/**
   2+1+1 Mixed Precision EOFA Mobius with a tonne of smearing

   Idea is to only do one EOFA ~det(c)/det(s) and a two flavor (det(PV)/det(c))^2 on top of the light quark Hasenbusch

   We don't do an explicit c quark but it comes in through the quotient and cutoff in the strange, which has major benefits as the
   EOFA is not cheap!!
 **/
#include <Grid/Grid.h>

NAMESPACE_BEGIN(Grid);

//#define b4008
#define b4068
//#define b416
//#define b4238
//#define b4300

//#define L32
#define L24
//#define L16

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
#if defined (b4008) || defined (b4068)
  CPparams.saveInterval  = 1;
#else
  CPparams.saveInterval  = 5;
#endif
  CPparams.saveSmeared   = false ;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  // these are initial seeds and then the file IO takes precedence
  RNGModuleParameters RNGpar;
  for( int i = 0 ; i < 5 ; i++ ) {
    RNGpar.serial_seeds += " " + std::to_string(HMCparams.Seed+i) ;
    RNGpar.parallel_seeds += " " + std::to_string(HMCparams.Seed+5+i) ;
  }
  std::cout<<"Serial Seeds"<<RNGpar.serial_seeds<<std::endl ;
  std::cout<<"Parallel Seeds"<<RNGpar.parallel_seeds<<std::endl ;
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  // Construct observables
  // here there is too much indirection 
  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  //////////////////////////////////////////////

#ifdef b4008
  const int Ls            = 10;
  const Real beta         = 4.008;
  const Real strange_mass = 0.0725;
  const Real charm_mass   = 11.8*strange_mass ;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.75; 
  const RealD c           = 0.75;
  #ifdef L32
    const Real light_mass   = 0.004;
    std::vector<Real> hasenbusch( { 0.0075, 0.0125, 0.0225, 0.0475, 0.09, 0.18, 0.36, 0.64 } ) ;
  #elif (defined L24)
    const Real light_mass   = 0.009;
    std::vector<Real> hasenbusch( { 0.016, 0.028, 0.045, 0.09, 0.18, 0.4, 0.64 } ) ;
  #elif (defined L16)
    const Real light_mass   = 0.019;
    std::vector<Real> hasenbusch( { 0.038, 0.09, 0.15, 0.3, 0.5 } ) ;
  #else
    #error "L not supported"
  #endif
  std::cout<<"aml,ams,amx = " << light_mass << " , " << strange_mass << " , " << charm_mass << std::endl ;
#elif (defined b4068)
  const int Ls            = 8;
  const Real beta         = 4.068;
  const Real strange_mass = 0.056;
  const Real charm_mass   = 11.8*strange_mass ;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.5; 
  const RealD c           = 0.5;
  #ifdef L32
    const Real light_mass   = 0.005 ;
    std::vector<Real> hasenbusch( { 0.013, 0.03, 0.06, 0.17, 0.33, 0.63 } ) ;
  #elif (defined L24)
    const Real light_mass   = 0.010 ;
    std::vector<Real> hasenbusch( { 0.017, 0.035, 0.07, 0.17, 0.33, 0.63 } ) ;
  #elif (defined L16)
    const Real light_mass   = 0.022 ;
    std::vector<Real> hasenbusch( { 0.04, 0.07, 0.17, 0.33, 0.61 } ) ;  
  #else
    #error "L not supported for these parameters"
  #endif
  std::cout<<"aml,ams,amx = " << light_mass << " , " << strange_mass << " , " << charm_mass << std::endl ;
    ///   
#elif (defined b416)
  const int Ls            = 6;
  const Real beta         = 4.160;
  const Real strange_mass = 0.042;
  const Real charm_mass   = 11.8*strange_mass ;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.5;
  const RealD c           = 0.5;
  #ifdef L32
    const Real light_mass   = 0.006;
    std::vector<Real> hasenbusch( { 0.02, 0.05, 0.15, 0.5 } ) ;
  #elif (defined L24)
    const Real light_mass   = 0.012;
    std::vector<Real> hasenbusch( { 0.05, 0.15, 0.5 } ) ;
  #else
    #error "Chosen L not supported"
  #endif
  std::cout<<"aml,ams,amx = " << light_mass << " , " << strange_mass << " , " << charm_mass << std::endl ;
#elif (defined b4238)
  const int Ls            = 4;
  const Real beta         = 4.238;
  const Real strange_mass = 0.032;
  const Real charm_mass   = 11.8*strange_mass ;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.25;
  const RealD c           = 0.25;
  #ifdef L32
    const Real light_mass   = 0.008;
    std::vector<Real> hasenbusch( { 0.035, 0.14, 0.4 } ) ;  
  #elif (defined L24)
    const Real light_mass   = 0.0145;
    std::vector<Real> hasenbusch( { 0.055, 0.14, 0.4 } ) ;  
  #else
    #error "L not supported"
  #endif 
  std::cout<<"aml,ams,amx = " << light_mass << " , " << strange_mass << " , " << charm_mass << std::endl ;
#elif (defined b4300)
  const int Ls            = 4;
  const Real beta         = 4.3;
  const Real strange_mass = 0.023;
  const Real charm_mass   = 11.8*strange_mass ;
  const Real pv_mass      = 1.0;
  const RealD M5          = 1.0;
  const RealD b           = 1.0;
  const RealD c           = 0.0;
  #ifdef L32
    const Real light_mass   = 0.0085;
    std::vector<Real> hasenbusch( { 0.03, 0.12, 0.35 } ) ;  
  #else
    #error "L not supported"
  #endif
  std::cout<<"aml,ams,amx = " << light_mass << " , " << strange_mass << " , " << charm_mass << std::endl ;
#else
  exit(1) ;
#endif
  
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
  
  ////////////////////////////////////
  // Strange/Charm action
  ////////////////////////////////////
  OneFlavourRationalParams OFRp;
  OFRp.lo        = 0.9;
  OFRp.hi        = 3.5;
  OFRp.MaxIter   = 10000;
  OFRp.tolerance = 1.0e-9;
  OFRp.degree    = 3 ;
  OFRp.precision = 50;

  ConjugateGradient<FermionField> ActionCG(ActionStoppingCondition,MaxCGIterations);

  // could put an intermediate hasenbusch here I suppose ....
#if (defined b4008)
  #if (defined L32) || (defined L16)
    std::vector<double> EOFAhs = { strange_mass , 0.18 , charm_mass } ;
  #elif (defined L24)
    std::vector<double> EOFAhs = { strange_mass , 0.2 , charm_mass } ;
  #endif
#elif (defined b416) || (defined b4068) || (defined b4238) || (defined b4300)
  std::vector<double> EOFAhs = { strange_mass , charm_mass } ;
#endif
  std::vector<MobiusEOFAFermionD*> Strange_Op_L  , Strange_Op_R  ;
  std::vector<MobiusEOFAFermionF*> Strange_Op_LF , Strange_Op_RF ;
  
  std::vector<LinearOperatorEOFAD*> Strange_LinOp_L  , Strange_LinOp_R  ;
  std::vector<LinearOperatorEOFAF*> Strange_LinOp_LF , Strange_LinOp_RF ;

  std::vector<MxPCG_EOFA*> ActionCGL , DerivativeCGL , ActionCGR , DerivativeCGR ;
  std::vector<ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> *> EOFA ;

  for( int i = 0 ; i < EOFAhs.size()-1 ; i++ ) {

    Strange_Op_L.push_back( new MobiusEOFAFermionD(U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , EOFAhs[i] , EOFAhs[i],  EOFAhs[i+1], 0.0, -1, M5, b, c) );
    Strange_Op_R.push_back( new MobiusEOFAFermionD(U , *FGrid , *FrbGrid , *GridPtr , *GridRBPtr , EOFAhs[i+1], EOFAhs[i], EOFAhs[i+1], -1.0, 1, M5, b, c) );
			    
    Strange_Op_LF.push_back( new MobiusEOFAFermionF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, EOFAhs[i] , EOFAhs[i],  EOFAhs[i+1], 0.0, -1, M5, b, c) );
    Strange_Op_RF.push_back( new MobiusEOFAFermionF(UF, *FGridF, *FrbGridF, *GridPtrF, *GridRBPtrF, EOFAhs[i+1], EOFAhs[i], EOFAhs[i+1], -1.0, 1, M5, b, c) );

    // Mixed precision EOFA
    Strange_LinOp_L.push_back( new LinearOperatorEOFAD( *Strange_Op_L[i] ) ) ;
    Strange_LinOp_R.push_back( new LinearOperatorEOFAD( *Strange_Op_R[i] ) ) ;
    Strange_LinOp_LF.push_back( new LinearOperatorEOFAF( *Strange_Op_LF[i] ) ) ;
    Strange_LinOp_RF.push_back( new LinearOperatorEOFAF( *Strange_Op_RF[i] ) ) ;

    ActionCGL.push_back( new MxPCG_EOFA( ActionStoppingCondition, MX_inner, MaxCGIterations, GridPtrF, FrbGridF, *Strange_Op_LF[i],*Strange_Op_L[i], *Strange_LinOp_LF[i],*Strange_LinOp_L[i]));
    DerivativeCGL.push_back( new MxPCG_EOFA( DerivativeStoppingCondition, MX_inner, MaxCGIterations, GridPtrF, FrbGridF, *Strange_Op_LF[i],*Strange_Op_L[i], *Strange_LinOp_LF[i],*Strange_LinOp_L[i]));
    ActionCGR.push_back( new MxPCG_EOFA( ActionStoppingCondition, MX_inner, MaxCGIterations, GridPtrF, FrbGridF, *Strange_Op_RF[i],*Strange_Op_R[i], *Strange_LinOp_RF[i],*Strange_LinOp_R[i]));
    DerivativeCGR.push_back( new MxPCG_EOFA( DerivativeStoppingCondition, MX_inner, MaxCGIterations, GridPtrF, FrbGridF, *Strange_Op_RF[i],*Strange_Op_R[i], *Strange_LinOp_RF[i],*Strange_LinOp_R[i]));

    EOFA.push_back( new ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy>( *Strange_Op_L[i], *Strange_Op_R[i], ActionCG, *ActionCGL[i], *ActionCGR[i], *DerivativeCGL[i], *DerivativeCGR[i], OFRp, true ) );
    EOFA[i] -> is_smeared = true ;

    // put them all on Level1 because the EOFA is pretty well behaved
#ifdef b4008
    if( i > 0 ) {
      Level2.push_back( EOFA[i] );
    } else {
      Level1.push_back( EOFA[i] );
    }
#elif (defined b4068)
    Level2.push_back( EOFA[i] );
#elif (defined b416)
    #ifdef L32
    Level2.push_back( EOFA[i] );
    #else
    Level1.push_back( EOFA[i] );
    #endif
#elif (defined b4238) || (defined b4300)
    Level1.push_back( EOFA[i] );
#endif
  }

  ////////////////////////////////////
  // up down action
  ////////////////////////////////////
  std::vector<Real> light_den , light_num;

  // charm as 2 flavor so we can EOFA the strange .... (c/s)(PV/c)^2 -> s/PV,c/PV and then the usual chain?

  int n_hasenbusch = hasenbusch.size();
  light_den.push_back(light_mass);
  for(int h=0;h<n_hasenbusch;h++){
    light_den.push_back(hasenbusch[h]);
    light_num.push_back(hasenbusch[h]);
  }
  // and then 
  light_num.push_back(pv_mass);
  
  // extra 2f charm here
  light_den.push_back( charm_mass ) ;
  light_num.push_back( pv_mass ) ;

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

  for(int h=0;h<n_hasenbusch+2;h++){

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
    MPCG.push_back(new MxPCG( conv,
			      MX_inner,
			      MaxCGIterations,
			      GridPtrF,
			      FrbGridF,
			      *DenominatorsF[h],*Denominators[h],
			      *LinOpF[h], *LinOpD[h]) );
    
    ActionMPCG.push_back(new MxPCG( ActionStoppingCondition,
				    MX_inner,
				    MaxCGIterations,
				    GridPtrF,
				    FrbGridF,
				    *DenominatorsF[h],*Denominators[h],
				    *LinOpF[h], *LinOpD[h]) );

    // Heatbath not mixed yet. As inverts numerators not so important as raised mass.
    Quotients.push_back (new TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicy>(*Numerators[h],*Denominators[h],*MPCG[h],*ActionMPCG[h],ActionCG));
    Quotients[h] -> is_smeared = true ;

    // put everything apart from the light quark on level 2
    #ifdef b4008
      #ifdef L32
        if( h > 3 ) {
      #elif (defined L24)
        if( h > 2 ) {
      #elif (defined L16)
        if( h > 1 ) {
      #endif
    #elif (defined b4068)
      if( h > 2 ) {
    #elif (defined b416)
      if( h > 1 ) {
    #elif (defined b4238) || (defined b4300)
    if( h > 0 ) {
    #endif
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
