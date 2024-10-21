/**
   2+1+1 Mixed Precision EOFA Mobius with a tonne of smearing
 **/
#include <Grid/Grid.h>

#define MIXED_PRECISION

#define EOFA_CHARM
#define EOFA_STRANGE

NAMESPACE_BEGIN(Grid);

  /*
   * Need a plan for gauge field update for mixed precision in HMC                      (2x speed up)
   *    -- Store the single prec action operator.
   *    -- Clone the gauge field from the operator function argument.
   *    -- Build the mixed precision operator dynamically from the passed operator and single prec clone.
   */

  template<class FermionOperatorD, class FermionOperatorF, class SchurOperatorD, class  SchurOperatorF> 
  class MixedPrecisionConjugateGradientOperatorFunction : public OperatorFunction<typename FermionOperatorD::FermionField> {
  public:
    typedef typename FermionOperatorD::FermionField FieldD;
    typedef typename FermionOperatorF::FermionField FieldF;

    using OperatorFunction<FieldD>::operator();

    RealD   Tolerance;
    RealD   InnerTolerance; //Initial tolerance for inner CG. Defaults to Tolerance but can be changed
    Integer MaxInnerIterations;
    Integer MaxOuterIterations;
    GridBase* SinglePrecGrid4; //Grid for single-precision fields
    GridBase* SinglePrecGrid5; //Grid for single-precision fields
    RealD OuterLoopNormMult; //Stop the outer loop and move to a final double prec solve when the residual is OuterLoopNormMult * Tolerance

    FermionOperatorF &FermOpF;
    FermionOperatorD &FermOpD;;
    SchurOperatorF &LinOpF;
    SchurOperatorD &LinOpD;

    Integer TotalInnerIterations; //Number of inner CG iterations
    Integer TotalOuterIterations; //Number of restarts
    Integer TotalFinalStepIterations; //Number of CG iterations in final patch-up step

    MixedPrecisionConjugateGradientOperatorFunction(RealD tol, 
						    Integer maxinnerit, 
						    Integer maxouterit, 
						    GridBase* _sp_grid4, 
						    GridBase* _sp_grid5, 
						    FermionOperatorF &_FermOpF,
						    FermionOperatorD &_FermOpD,
						    SchurOperatorF   &_LinOpF,
						    SchurOperatorD   &_LinOpD): 
      LinOpF(_LinOpF),
      LinOpD(_LinOpD),
      FermOpF(_FermOpF),
      FermOpD(_FermOpD),
      Tolerance(tol), 
      InnerTolerance(tol), 
      MaxInnerIterations(maxinnerit), 
      MaxOuterIterations(maxouterit), 
      SinglePrecGrid4(_sp_grid4),
      SinglePrecGrid5(_sp_grid5),
      OuterLoopNormMult(100.) 
    { 
    };

    void operator()(LinearOperatorBase<FieldD> &LinOpU, const FieldD &src, FieldD &psi) {

      std::cout << GridLogMessage << " Mixed precision CG wrapper operator() "<<std::endl;

      SchurOperatorD * SchurOpU = static_cast<SchurOperatorD *>(&LinOpU);
      assert(&(SchurOpU->_Mat)==&(LinOpD._Mat));

      ////////////////////////////////////////////////////////////////////////////////////
      // Must snarf a single precision copy of the gauge field in Linop_d argument
      ////////////////////////////////////////////////////////////////////////////////////
      typedef typename FermionOperatorF::GaugeField GaugeFieldF;
      typedef typename FermionOperatorF::GaugeLinkField GaugeLinkFieldF;
      typedef typename FermionOperatorD::GaugeField GaugeFieldD;
      typedef typename FermionOperatorD::GaugeLinkField GaugeLinkFieldD;

      GridBase * GridPtrF = SinglePrecGrid4;
      GridBase * GridPtrD = FermOpD.Umu.Grid();
      GaugeFieldF     U_f  (GridPtrF);
      GaugeLinkFieldF Umu_f(GridPtrF);
      
      ////////////////////////////////////////////////////////////////////////////////////
      // Moving this to a Clone method of fermion operator would allow to duplicate the 
      // physics parameters and decrease gauge field copies
      ////////////////////////////////////////////////////////////////////////////////////
      GaugeLinkFieldD Umu_d(GridPtrD);
      for(int mu=0;mu<Nd*2;mu++){ 
	Umu_d = PeekIndex<LorentzIndex>(FermOpD.Umu, mu);
	precisionChange(Umu_f,Umu_d);
	PokeIndex<LorentzIndex>(FermOpF.Umu, Umu_f, mu);
      }
      pickCheckerboard(Even,FermOpF.UmuEven,FermOpF.Umu);
      pickCheckerboard(Odd ,FermOpF.UmuOdd ,FermOpF.Umu);

      ////////////////////////////////////////////////////////////////////////////////////
      // Make a mixed precision conjugate gradient
      ////////////////////////////////////////////////////////////////////////////////////
      MixedPrecisionConjugateGradient<FieldD,FieldF> MPCG(Tolerance,MaxInnerIterations,MaxOuterIterations,SinglePrecGrid5,LinOpF,LinOpD);
      std::cout << GridLogMessage << "Calling mixed precision Conjugate Gradient" <<std::endl;
      MPCG(src,psi);
    }
  };

NAMESPACE_END(Grid);


int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  // here make a routine to print all the relevant information on the run
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

   // Typedefs to simplify notation
  typedef WilsonImplR FermionImplPolicy;
  typedef MobiusFermionD FermionAction;
  typedef MobiusFermionF FermionActionF;
  typedef MobiusEOFAFermionD FermionEOFAAction;
  typedef MobiusEOFAFermionF FermionEOFAActionF;
  typedef typename FermionAction::FermionField FermionField;
  typedef typename FermionActionF::FermionField FermionFieldF;

  typedef Grid::XmlReader       Serialiser;
  
  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  typedef GenericHMCRunner<Integrator> HMCWrapper;

  HMCparameters HMCparams;
  {
    XmlReader HMCrd("HMCparameters.xml");
    read(HMCrd,"HMCparameters",HMCparams);
    std::cout << GridLogMessage<< HMCparams <<std::endl;
  } 
  HMCWrapper TheHMC(HMCparams);

  // Grid from the command line arguments --grid and --mpi
  TheHMC.Resources.AddFourDimGrid("gauge"); // use default simd lanes decomposition
  
  CheckpointerParameters CPparams;
  CPparams.config_prefix = "ckpoint_EODWF_lat";
  CPparams.rng_prefix    = "ckpoint_EODWF_rng";
  CPparams.saveInterval  = 1;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  // Construct observables
  // here there is too much indirection 
  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  //////////////////////////////////////////////

  const int Ls            = 10;
  const Real beta         = 4.09;
  const Real light_mass   = 0.010;
  const Real strange_mass = 0.056;
  const Real charm_mass   = 0.630;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.5; 
  const RealD c           = 0.5;

  std::vector<Real> hasenbusch( { 0.06, 0.4 } ) ;  

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  auto FGrid     = SpaceTimeGrid::makeFiveDimGrid(Ls,GridPtr);
  auto FrbGrid   = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,GridPtr);

  Coordinate latt  = GridDefaultLatt();
  Coordinate mpi   = GridDefaultMpi();
  Coordinate simdF = GridDefaultSimd(Nd,vComplexF::Nsimd());
  Coordinate simdD = GridDefaultSimd(Nd,vComplexD::Nsimd());
  auto GridPtrF    = SpaceTimeGrid::makeFourDimGrid(latt,simdF,mpi);
  auto GridRBPtrF  = SpaceTimeGrid::makeFourDimRedBlackGrid(GridPtrF);
  auto FGridF      = SpaceTimeGrid::makeFiveDimGrid(Ls,GridPtrF);
  auto FrbGridF    = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,GridPtrF);

  // temporarily need a gauge field
  LatticeGaugeField U(GridPtr);
  LatticeGaugeFieldF UF(GridPtrF);

  // These lines are unecessary if BC are all periodic
  std::vector<Complex> boundary = {1,1,1,-1};
  FermionAction::ImplParams Params(boundary);
  FermionActionF::ImplParams ParamsF(boundary);
  
  const double ActionStoppingCondition     = 1e-10;
  const double DerivativeStoppingCondition = 1e-7;
  const double MaxCGIterations             = 30000;

  ////////////////////////////////////
  // Collect actions
  ////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1( HMCparams.MD.lvl_sizes[0] , IntStringToEnum( HMCparams.MD.name[0] ) );
  ActionLevel<HMCWrapper::Field> Level2( HMCparams.MD.lvl_sizes[1] , IntStringToEnum( HMCparams.MD.name[1] ) );
  ActionLevel<HMCWrapper::Field> Level3( HMCparams.MD.lvl_sizes[2] , IntStringToEnum( HMCparams.MD.name[2] ) );

  typedef SchurDiagMooeeOperator<FermionActionF,FermionFieldF> LinearOperatorF;
  typedef SchurDiagMooeeOperator<FermionAction ,FermionField > LinearOperatorD;
  typedef SchurDiagMooeeOperator<FermionEOFAActionF,FermionFieldF> LinearOperatorEOFAF;
  typedef SchurDiagMooeeOperator<FermionEOFAAction ,FermionField > LinearOperatorEOFAD;

  typedef MixedPrecisionConjugateGradientOperatorFunction<MobiusFermionD,MobiusFermionF,LinearOperatorD,LinearOperatorF> MxPCG;
  typedef MixedPrecisionConjugateGradientOperatorFunction<MobiusEOFAFermionD,MobiusEOFAFermionF,LinearOperatorEOFAD,LinearOperatorEOFAF> MxPCG_EOFA;

  ////////////////////////////////////
  // Charm action -- should we put this on level 2??? Maybe
  ////////////////////////////////////
  const int MX_inner = 5000;

#ifdef EOFA_CHARM
  OneFlavourRationalParams Charm_OFRp ;
  Charm_OFRp.lo = 1.0 ;
  Charm_OFRp.hi = 3.5 ;
  Charm_OFRp.MaxIter = 10000 ;
  Charm_OFRp.tolerance = 1E-9 ;
  Charm_OFRp.degree = 4 ;
  Charm_OFRp.precision = 50 ;

  MobiusEOFAFermionD Charm_Op_L (U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , charm_mass, charm_mass, pv_mass, 0.0, -1, M5, b, c);
  MobiusEOFAFermionF Charm_Op_LF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, charm_mass, charm_mass, pv_mass, 0.0, -1, M5, b, c);
  MobiusEOFAFermionD Charm_Op_R (U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , pv_mass, charm_mass,    pv_mass, -1.0, 1, M5, b, c);
  MobiusEOFAFermionF Charm_Op_RF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, pv_mass, charm_mass,    pv_mass, -1.0, 1, M5, b, c);

  ConjugateGradient<FermionField> Charm_ActionCG(1E-14,MaxCGIterations);
  ConjugateGradient<FermionField> Charm_DerivativeCG(1E-10,MaxCGIterations);

  // Mixed precision EOFA
  LinearOperatorEOFAD Charm_LinOp_L (Charm_Op_L);
  LinearOperatorEOFAD Charm_LinOp_R (Charm_Op_R);
  LinearOperatorEOFAF Charm_LinOp_LF(Charm_Op_LF);
  LinearOperatorEOFAF Charm_LinOp_RF(Charm_Op_RF);

  MxPCG_EOFA Charm_ActionCGL(1E-14,
			     MX_inner,
			     MaxCGIterations,
			     GridPtrF,
			     FrbGridF,
			     Charm_Op_LF,Charm_Op_L,
			     Charm_LinOp_LF,Charm_LinOp_L);

  MxPCG_EOFA Charm_DerivativeCGL(DerivativeStoppingCondition,
				 MX_inner,
				 MaxCGIterations,
				 GridPtrF,
				 FrbGridF,
				 Charm_Op_LF,Charm_Op_L,
				 Charm_LinOp_LF,Charm_LinOp_L);
  
  MxPCG_EOFA Charm_ActionCGR(ActionStoppingCondition,
			     MX_inner,
			     MaxCGIterations,
			     GridPtrF,
			     FrbGridF,
			     Charm_Op_RF,Charm_Op_R,
			     Charm_LinOp_RF,Charm_LinOp_R);
  
  MxPCG_EOFA Charm_DerivativeCGR(DerivativeStoppingCondition,
				 MX_inner,
				 MaxCGIterations,
				 GridPtrF,
				 FrbGridF,
				 Charm_Op_RF,Charm_Op_R,
				 Charm_LinOp_RF,Charm_LinOp_R);
  
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> 
    Charm_EOFA(Charm_Op_L, Charm_Op_R, 
	       Charm_ActionCG, 
	       Charm_ActionCGL, Charm_ActionCGR,
	       Charm_DerivativeCGL, Charm_DerivativeCGR,
	       Charm_OFRp, true);
#else
  OneFlavourRationalParams Charm_OFRp;
  Charm_OFRp.lo       = 0.1;
  Charm_OFRp.hi       = 3.5;
  Charm_OFRp.MaxIter  = 10000;
  Charm_OFRp.tolerance= 1.0e-10;
  Charm_OFRp.degree   = 5;
  Charm_OFRp.precision= 50;

  FermionAction CharmOp (U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,charm_mass,M5,b,c, Params);
  FermionAction CharmPauliVillarsOp(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,pv_mass,  M5,b,c, Params);

  OneFlavourEvenOddRatioRationalPseudoFermionAction<FermionImplPolicy> \
    Charm_EOFA(CharmPauliVillarsOp,CharmOp,Charm_OFRp);
#endif
  Charm_EOFA.is_smeared = true ;
  Level1.push_back(&Charm_EOFA);
  
  ////////////////////////////////////
  // Strange action
  ////////////////////////////////////
#ifdef EOFA_STRANGE
  // DJM: setup for EOFA ratio (Mobius)
  OneFlavourRationalParams OFRp;
  OFRp.lo       = 0.75;
  OFRp.hi       = 4.5;
  OFRp.MaxIter  = 10000;
  OFRp.tolerance= 1.0e-9;
  OFRp.degree   = 5 ;
  OFRp.precision= 50;

  ConjugateGradient<FermionField> ActionCG(ActionStoppingCondition,MaxCGIterations);
  ConjugateGradient<FermionField> DerivativeCG(DerivativeStoppingCondition,MaxCGIterations);
  
  MobiusEOFAFermionD Strange_Op_L (U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , strange_mass, strange_mass, pv_mass, 0.0, -1, M5, b, c);
  MobiusEOFAFermionF Strange_Op_LF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, strange_mass, strange_mass, pv_mass, 0.0, -1, M5, b, c);
  MobiusEOFAFermionD Strange_Op_R (U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , pv_mass, strange_mass,      pv_mass, -1.0, 1, M5, b, c);
  MobiusEOFAFermionF Strange_Op_RF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, pv_mass, strange_mass,      pv_mass, -1.0, 1, M5, b, c);

  // Mixed precision EOFA
  LinearOperatorEOFAD Strange_LinOp_L (Strange_Op_L);
  LinearOperatorEOFAD Strange_LinOp_R (Strange_Op_R);
  LinearOperatorEOFAF Strange_LinOp_LF(Strange_Op_LF);
  LinearOperatorEOFAF Strange_LinOp_RF(Strange_Op_RF);

  MxPCG_EOFA ActionCGL(ActionStoppingCondition,
		       MX_inner,
		       MaxCGIterations,
		       GridPtrF,
		       FrbGridF,
		       Strange_Op_LF,Strange_Op_L,
		       Strange_LinOp_LF,Strange_LinOp_L);

  MxPCG_EOFA DerivativeCGL(DerivativeStoppingCondition,
			   MX_inner,
			   MaxCGIterations,
			   GridPtrF,
			   FrbGridF,
			   Strange_Op_LF,Strange_Op_L,
			   Strange_LinOp_LF,Strange_LinOp_L);
  
  MxPCG_EOFA ActionCGR(ActionStoppingCondition,
		       MX_inner,
		       MaxCGIterations,
		       GridPtrF,
		       FrbGridF,
		       Strange_Op_RF,Strange_Op_R,
		       Strange_LinOp_RF,Strange_LinOp_R);
  
  MxPCG_EOFA DerivativeCGR(DerivativeStoppingCondition,
			   MX_inner,
			   MaxCGIterations,
			   GridPtrF,
			   FrbGridF,
			   Strange_Op_RF,Strange_Op_R,
			   Strange_LinOp_RF,Strange_LinOp_R);

  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> 
    EOFA(Strange_Op_L, Strange_Op_R, 
	 ActionCG, 
	 ActionCGL, ActionCGR,
	 DerivativeCGL, DerivativeCGR,
	 OFRp, true);
#else
  OneFlavourRationalParams OFRp;
  OFRp.lo       = 0.2;
  OFRp.hi       = 4.5;
  OFRp.MaxIter  = 10000;
  OFRp.tolerance= 1.0e-10;
  OFRp.degree   = 3;
  OFRp.precision= 50;

  FermionAction StrangeOp (U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,strange_mass,M5,b,c, Params);
  FermionAction StrangePauliVillarsOp(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,pv_mass,  M5,b,c, Params);

  OneFlavourEvenOddRatioRationalPseudoFermionAction<FermionImplPolicy> \
    EOFA(StrangePauliVillarsOp,StrangeOp,OFRp);
#endif
  Level2.push_back(&EOFA);
  EOFA.is_smeared = true ;

  ////////////////////////////////////
  // up down action
  ////////////////////////////////////
  std::vector<Real> light_den;
  std::vector<Real> light_num;

  int n_hasenbusch = hasenbusch.size();
  light_den.push_back(light_mass);
  for(int h=0;h<n_hasenbusch;h++){
    light_den.push_back(hasenbusch[h]);
    light_num.push_back(hasenbusch[h]);
  }
  light_num.push_back(pv_mass);

  //////////////////////////////////////////////////////////////
  // Forced to replicate the MxPCG and DenominatorsF etc.. because
  // there is no convenient way to "Clone" physics params from double op
  // into single op for any operator pair.
  // Same issue prevents using MxPCG in the Heatbath step
  //////////////////////////////////////////////////////////////
  std::vector<FermionAction *> Numerators;
  std::vector<FermionAction *> Denominators;
  std::vector<TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy> *> Quotients;
  std::vector<MxPCG *> ActionMPCG;
  std::vector<MxPCG *> MPCG;
  std::vector<FermionActionF *> DenominatorsF;
  std::vector<LinearOperatorD *> LinOpD;
  std::vector<LinearOperatorF *> LinOpF; 

  for(int h=0;h<n_hasenbusch+1;h++){

    std::cout << GridLogMessage << " 2f quotient Action  "<< light_num[h] << " / " << light_den[h]<< std::endl;

    Numerators.push_back  (new FermionAction(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,light_num[h],M5,b,c, Params));
    Denominators.push_back(new FermionAction(U,*FGrid,*FrbGrid,*GridPtr,*GridRBPtr,light_den[h],M5,b,c, Params));

    ////////////////////////////////////////////////////////////////////////////
    // Mixed precision CG for 2f force
    ////////////////////////////////////////////////////////////////////////////
    double DerivativeStoppingConditionLoose = 3e-7;

    DenominatorsF.push_back(new FermionActionF(UF,*FGridF,*FrbGridF,*GridPtrF,*GridRBPtrF,light_den[h],M5,b,c, ParamsF));
    LinOpD.push_back(new LinearOperatorD(*Denominators[h]));
    LinOpF.push_back(new LinearOperatorF(*DenominatorsF[h]));

    double conv  = DerivativeStoppingCondition;
    if (h<3) conv= DerivativeStoppingConditionLoose; // Relax on first two hasenbusch factors
    MPCG.push_back(new MxPCG(conv,
			     MX_inner,
			     MaxCGIterations,
			     GridPtrF,
			     FrbGridF,
			     *DenominatorsF[h],*Denominators[h],
			     *LinOpF[h], *LinOpD[h]) );

    ActionMPCG.push_back(new MxPCG(ActionStoppingCondition,
				   MX_inner,
				   MaxCGIterations,
				   GridPtrF,
				   FrbGridF,
				   *DenominatorsF[h],*Denominators[h],
				   *LinOpF[h], *LinOpD[h]) );

    // Heatbath not mixed yet. As inverts numerators not so important as raised mass.
    Quotients.push_back (new TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy>(*Numerators[h],*Denominators[h],*MPCG[h],*ActionMPCG[h],ActionCG));
  }

  for(int h=0;h<n_hasenbusch+1;h++){
    Quotients[h] -> is_smeared = true ;
    if( h > 0 ) {
      Level2.push_back(Quotients[h]);
    } else {
      Level1.push_back(Quotients[h]);
    }
  }

  /////////////////////////////////////////////////////////////
  // Gauge action
  /////////////////////////////////////////////////////////////

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  SymanzikGaugeActionR GaugeAction(beta);
  GaugeAction.is_smeared = false ;
  Level3.push_back(&GaugeAction);
  
  TheHMC.TheAction.push_back(Level3);
  std::cout << GridLogMessage << " Action complete "<< std::endl;

  /////////////////////////////////////////////////////////////
  // HMC parameters are serialisable

  //SmearingParameters SmPar(Reader);
  double rho = 0.125;  // smearing parameter
  int Nsmear = 8;    // number of smearing levels
  Smear_Stout<HMCWrapper::ImplPolicy> Stout(rho);
  SmearedConfiguration<HMCWrapper::ImplPolicy> SmearingPolicy(GridPtr, Nsmear, Stout);

  std::cout << GridLogMessage << " Running the HMC "<< std::endl;
  TheHMC.Run(SmearingPolicy); // for smearing

  Grid_finalize();
} // main
