/*!
  @file GaugeConfiguration.h

  @brief Declares the GaugeConfiguration class
*/
#pragma once

NAMESPACE_BEGIN(Grid);


//trivial class for no smearing
template< class Impl >
class NoSmearing : public ConfigurationBase<typename Impl::Field>
{
public:
  INHERIT_FIELD_TYPES(Impl);

  Field* ThinLinks;

  NoSmearing(): ThinLinks(NULL) {}

  virtual void set_Field(Field& U) { ThinLinks = &U; }

  virtual void smeared_force(Field&) {}

  virtual Field& get_SmearedU() { return *ThinLinks; }

  virtual Field &get_U(bool smeared = false)
  {
    return *ThinLinks;
  }
};

/*!
  @brief Smeared configuration container

  It will behave like a configuration from the point of view of
  the HMC update and integrators.
  An "advanced configuration" object that can provide not only the
  data to store the gauge configuration but also operations to manipulate
  it, like smearing.

  It stores a list of smeared configurations.
*/
template <class Gimpl>
class SmearedConfiguration : public ConfigurationBase<typename Gimpl::Field>
{
public:
  INHERIT_GIMPL_TYPES(Gimpl);

protected:
  const unsigned int smearingLevels;
  Smear_Stout<Gimpl> *StoutSmearing;
  std::vector<GaugeField> SmearedSet;
public:
  GaugeField*  ThinLinks; /* Pointer to the thin links configuration */ // move to base???
protected:
  
  // Member functions
  //====================================================================

  // Overridden in masked version
  virtual void fill_smearedSet(GaugeField &U)
  {
    ThinLinks = &U;  // attach the smearing routine to the field U

    // check the pointer is not null
    if (ThinLinks == NULL)
      std::cout << GridLogError
                << "[SmearedConfiguration] Error in ThinLinks pointer\n";

    if (smearingLevels > 0)
    {
      std::cout << GridLogDebug
                << "[SmearedConfiguration] Filling SmearedSet\n";
      GaugeField previous_u(ThinLinks->Grid());

      previous_u = *ThinLinks;
      for (int smearLvl = 0; smearLvl < smearingLevels; ++smearLvl)
      {
        StoutSmearing->smear(SmearedSet[smearLvl], previous_u);
        previous_u = SmearedSet[smearLvl];

        // For debug purposes
        RealD impl_plaq = WilsonLoops<Gimpl>::avgPlaquette(previous_u);
        std::cout << GridLogDebug
                  << "[SmearedConfiguration] Plaq: " << impl_plaq << std::endl;
      }
    }
  }

  //overridden in masked verson
  virtual GaugeField AnalyticSmearedForce(const GaugeField& SigmaKPrime,
					  const GaugeField& GaugeK) const 
  {
    GridBase* grid = GaugeK.Grid();
    GaugeField C(grid), SigmaK(grid), iLambda(grid);
    GaugeLinkField iLambda_mu(grid);
    GaugeLinkField e_iQ(grid);
    GaugeLinkField SigmaKPrime_mu(grid);
    GaugeLinkField GaugeKmu(grid), Cmu(grid);
    StoutSmearing->BaseSmear(C, GaugeK);
    for (int mu = 0; mu < Nd; mu++) {
      Cmu = peekLorentz(C, mu);
      GaugeKmu = peekLorentz(GaugeK, mu);
      SigmaKPrime_mu = peekLorentz(SigmaKPrime, mu);
      set_iLambda(iLambda_mu, e_iQ, Ta(Cmu * adj(GaugeKmu)),
		  SigmaKPrime_mu, GaugeKmu);
      pokeLorentz(SigmaK, SigmaKPrime_mu * e_iQ + adj(Cmu) * iLambda_mu, mu);
      pokeLorentz(iLambda, iLambda_mu, mu);
    }
    StoutSmearing->derivative(SigmaK, iLambda, GaugeK);  // derivative of SmearBase
    return SigmaK;
  }

  /*! @brief Returns smeared configuration at level 'Level' */
  const GaugeField &get_smeared_conf(int Level) const
  {
    return SmearedSet[Level];
  }

  //====================================================================
  void set_iLambda(GaugeLinkField& iLambda,
		   GaugeLinkField& e_iQ,
                   const GaugeLinkField& iQ,
		   const GaugeLinkField& Sigmap,
                   const GaugeLinkField& GaugeK) const 
  {
    GridBase* grid = iQ.Grid();
    GaugeLinkField B1(grid), B2(grid) ;
    LatticeComplex LatticeUnitComplex(grid);
    LatticeUnitComplex = 1.0;

    GaugeLinkField iQ2 = iQ*iQ;

    // sign in c0 from the conventions on the Ta
    LatticeComplex u = -imag(trace(iQ2*iQ))*0.3333333333333333148 ;
    LatticeComplex w = -real(trace(iQ2))*0.5;
    LatticeComplex f0 = 0.3849001794597505244*w ;
    w = sqrt(w) ;
    f0 = f0*w ;
    f0 = acos(u/f0)*0.3333333333333333148;
    u = w*(0.5773502691896257311)*cos(f0);
    w = w*sin(f0);

    const LatticeComplex xi0 = StoutSmearing -> func_xi0(w);
    LatticeComplex f2 = timesI( xi0 );
    const LatticeComplex u2 = u * u;
    const LatticeComplex w2 = w * w;
    const LatticeComplex xi1 = StoutSmearing -> func_xi1(w);
    // set w to cos(w) as the actual value of w is not used after here
    w = cos(w);
    
    LatticeComplex emiu = cos(u) - timesI(sin(u));
    //LatticeComplex e2iu = adj(emiu)*adj(emiu) ; weirdly sensitive to this ....
    u *= 2. ;
    LatticeComplex e2iu = cos(u) + timesI(sin(u));
    
    f0 = e2iu * (u2 - w2) + emiu * ((8.0*u2 * w) + (u * (3.0*u2 + w2) * f2));
    LatticeComplex f1 = e2iu*u - emiu * ((u * w) - (3.0*u2 - w2) * f2);
    f2 = e2iu - emiu * (w + (1.5*u) * f2);
    
    const LatticeComplex r01 = (u + timesI(2.0*(u2 - w2))) * e2iu +
      emiu * ((8.*u*w + u * (3.0 * u2 + w2) * xi0) +
	      timesI(-8.0 * u2 * w + 2.0 * (9.0 * u2 + w2) * xi0));
    
    const LatticeComplex r11 = (2.0 * LatticeUnitComplex + timesI(2.0*u)) * e2iu +
      emiu * ((-2.0 * w + (3.0 * u2 - w2) * xi0) +
	      timesI(u*(w + 3.0*xi0)));
    
    const LatticeComplex r21 = 2.0*timesI(e2iu) + emiu * (-1.5*u * xi0 + timesI(w - 3.0 * xi0));
    
    const LatticeComplex r02 = -2.0 * e2iu +
      emiu * (-8.0*u2*xi0 + timesI(u * (w + xi0 + 3.0*u2*xi1)));

    const LatticeComplex r12 = emiu * (u * xi0 + timesI(-w - xi0 + 3.0*u2*xi1));

    const LatticeComplex r22 = emiu * (xi0 - timesI(1.5*u * xi1));
    
    w = 1.0 ; 
    w = w / (9.0 * u2 - w2);  // reals
    f0 = f0 * w ;
    f1 = f1 * w ;
    f2 = f2 * w ;

    w *= (0.5*w) ;
    
    // exponentiate B1
    emiu = 3.*u2 - w2 ;
    e2iu = 30.*u2 + 2.*w2 ;
    LatticeComplex b0 = u*r01 + emiu*r02 - e2iu*f0;
    LatticeComplex b1 = u*r11 + emiu*r12 - e2iu*f1;
    LatticeComplex b2 = u*r21 + emiu*r22 - e2iu*f2;
    b0 *= w;
    b1 *= w;
    b2 *= w;
    B1 = 1.0 ;
    B1 = b0*B1 + timesMinusI(b1)*iQ - b2*iQ2;

    // exponentiate B2
    emiu = 1.5*u ;
    e2iu = 12.*u ;
    b0 = r01 - emiu*r02 - e2iu*f0;
    b1 = r11 - emiu*r12 - e2iu*f1;
    b2 = r21 - emiu*r22 - e2iu*f2;
    b0 *= w;
    b1 *= w;
    b2 *= w;
    B2 = 1.0 ;
    B2 = b0*B2 + timesMinusI(b1)*iQ - b2*iQ2;

    // compute sigmap, which we reuse the space for e_iQ
    e_iQ = GaugeK * Sigmap;
    iLambda = Ta( trace(e_iQ * B1) * iQ - timesI( trace(e_iQ * B2) )*iQ2 +
		  timesI(f1) * e_iQ + f2 * (iQ * e_iQ + e_iQ * iQ) );

    // finally return the exponential
    e_iQ = 1.0 ;
    e_iQ = f0*e_iQ + timesMinusI(f1)*iQ - f2*iQ2;
  }

  //====================================================================
public:

  /* Standard constructor */
  SmearedConfiguration(GridCartesian* UGrid, unsigned int Nsmear,
                       Smear_Stout<Gimpl>& Stout)
      : smearingLevels(Nsmear), StoutSmearing(&Stout), ThinLinks(NULL)
  {
    for (unsigned int i = 0; i < smearingLevels; ++i)
      SmearedSet.push_back(*(new GaugeField(UGrid)));
  }

  /*! For just thin links */
  SmearedConfiguration()
    : smearingLevels(0), StoutSmearing(nullptr), SmearedSet(), ThinLinks(NULL) {}

  // attach the smeared routines to the thin links U and fill the smeared set
  virtual void set_Field(GaugeField &U)
  {
    double start = usecond();
    fill_smearedSet(U);
    double end = usecond();
    double time = (end - start)/ 1e3;
    std::cout << GridLogMessage << "Smearing in " << time << " ms" << std::endl;  
  }

  //====================================================================
  virtual void smeared_force(GaugeField &SigmaTilde) 
  {
    if (smearingLevels > 0)
    {
      double start = usecond();
      GaugeField force = SigmaTilde; // actually = U*SigmaTilde
      GaugeLinkField tmp_mu(SigmaTilde.Grid());

      for (int mu = 0; mu < Nd; mu++)
      {
        // to get just SigmaTilde
        tmp_mu = adj(peekLorentz(SmearedSet[smearingLevels - 1], mu)) * peekLorentz(force, mu);
        pokeLorentz(force, tmp_mu, mu);
      }

      for (int ismr = smearingLevels - 1; ismr > 0; --ismr)
        force = AnalyticSmearedForce(force, get_smeared_conf(ismr - 1));
      force = AnalyticSmearedForce(force, *ThinLinks);

      for (int mu = 0; mu < Nd; mu++)
      {
        tmp_mu = peekLorentz(*ThinLinks, mu) * peekLorentz(force, mu);
        pokeLorentz(SigmaTilde, tmp_mu, mu);
      }
      double end = usecond();
      double time = (end - start)/ 1e3;
      std::cout << GridLogMessage << " GaugeConfiguration: Smeared Force chain rule took " << time << " ms" << std::endl;
    }  // if smearingLevels = 0 do nothing
    SigmaTilde=Gimpl::projectForce(SigmaTilde); // Ta
      
  }
  //====================================================================

  virtual GaugeField& get_SmearedU() { return SmearedSet[smearingLevels - 1]; }

  virtual GaugeField &get_U(bool smeared = false)
  {
    // get the config, thin links by default
    if (smeared)
    {
      if (smearingLevels)
      {
        RealD impl_plaq =
	  WilsonLoops<Gimpl>::avgPlaquette(SmearedSet[smearingLevels - 1]);
        std::cout << GridLogDebug << "getting Usmr Plaq: " << impl_plaq
                  << std::endl;
        return get_SmearedU();
      }
      else
      {
        RealD impl_plaq = WilsonLoops<Gimpl>::avgPlaquette(*ThinLinks);
        std::cout << GridLogDebug << "getting Thin Plaq: " << impl_plaq
                  << std::endl;
        return *ThinLinks;
      }
    }
    else
    {
      RealD impl_plaq = WilsonLoops<Gimpl>::avgPlaquette(*ThinLinks);
      std::cout << GridLogDebug << "getting Thin Plaq: " << impl_plaq
                << std::endl;
      return *ThinLinks;
    }
  }
};

NAMESPACE_END(Grid);

