#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

static inline LatticeComplex func_xi0(const LatticeComplex& w) {
  return sin(w) / w;
}

static inline LatticeComplex func_xi1(const LatticeComplex& w) {
  return cos(w) / (w * w) - sin(w) / (w * w * w);
}

static void set_uw_new(LatticeComplex& u,
		       LatticeComplex& w,
		       const LatticeColourMatrix &iQ2,
		       const LatticeColourMatrix &iQ3)
{
  // sign in c0 from the conventions on the Ta
  u = -imag(trace(iQ3))*0.3333333333333333148 ;
  w = -real(trace(iQ2))*0.5;

  LatticeComplex c0max = 0.3849001794597505244*w ;
  w = sqrt(w) ;
  c0max = c0max*w ;

  c0max = acos(u/c0max)*0.3333333333333333148;
  u = w*(0.5773502691896257311)*cos(c0max);
  w = w*sin(c0max);
}

static void set_uw_old(LatticeComplex& u,
		       LatticeComplex& w,
		       const LatticeColourMatrix &iQ2,
		       const LatticeColourMatrix &iQ3)
{
  Complex one_over_three = 1.0 / 3.0;
  Complex one_over_two = 1.0 / 2.0;

  GridBase* grid = u.Grid();
  LatticeComplex c0(grid), c1(grid), tmp(grid), c0max(grid), theta(grid);

  // sign in c0 from the conventions on the Ta
  c0 = -imag(trace(iQ3)) * one_over_three;  
  c1 = -real(trace(iQ2)) * one_over_two;

  // Cayley Hamilton checks to machine precision, tested
  tmp = c1 * one_over_three;
  c0max = 2.0 * pow(tmp, 1.5);

  theta = acos(c0 / c0max) *
    one_over_three;  // divide by three here, now leave as it is
  u = sqrt(tmp) * cos(theta);
  w = sqrt(c1) * sin(theta);
}

static void set_fj_old(LatticeComplex& f0,
		       LatticeComplex& f1,
		       LatticeComplex& f2,
		       const LatticeComplex& u,
		       const LatticeComplex& w)
{
  GridBase* grid = u.Grid();
  LatticeComplex xi0(grid), u2(grid), w2(grid), cosw(grid);
  LatticeComplex fden(grid);
  LatticeComplex h0(grid), h1(grid), h2(grid);
  LatticeComplex e2iu(grid), emiu(grid), ixi0(grid), qt(grid);
  LatticeComplex unity(grid);
  unity = 1.0;

  xi0 = func_xi0(w);
  u2 = u * u;
  w2 = w * w;
  cosw = cos(w);

  ixi0 = timesI(xi0);
  emiu = cos(u) - timesI(sin(u));
  e2iu = cos(2.0 * u) + timesI(sin(2.0 * u));

  h0 = e2iu * (u2 - w2) +
    emiu * ((8.0 * u2 * cosw) + (2.0 * u * (3.0 * u2 + w2) * ixi0));
  h1 = e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0);
  h2 = e2iu - emiu * (cosw + (3.0 * u) * ixi0);

  fden = unity / (9.0 * u2 - w2);  // reals
  f0 = h0 * fden;
  f1 = h1 * fden;
  f2 = h2 * fden;
}

static void set_fj_new(LatticeComplex& f0,
		       LatticeComplex& f1,
		       LatticeComplex& f2,
		       const LatticeComplex& u,
		       const LatticeComplex& w)
{
  const LatticeComplex u2 = u * u;
  const LatticeComplex w2 = w * w;
  const LatticeComplex cosw = cos(w);
  
  const LatticeComplex emiu = cos(u) - timesI(sin(u));
  // avoid another trig call
  //e2iu = cos(2.0 * u) + timesI(sin(2.0 * u));
  const LatticeComplex e2iu = adj(emiu)*adj(emiu) ;
  LatticeComplex ixi0 = timesI( func_xi0(w) );  
  f0 = e2iu * (u2 - w2) + emiu * ((8.0 * u2 * cosw) + (2.*u*(3.0 * u2 + w2) * ixi0));
  f1 = e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0);
  f2 = e2iu - emiu * (cosw + (3.0 * u) * ixi0);
  
  ixi0 = 1.0 ; 
  ixi0 = ixi0 / (9.0 * u2 - w2);  // reals
  f0 = f0 * ixi0 ;
  f1 = f1 * ixi0 ;
  f2 = f2 * ixi0 ;
}

static void
exponentiate_iQ_new( LatticeColourMatrix& e_iQ,
		     const LatticeColourMatrix& iQ)
{
#if (Config_Nc == 2)
  const LatticeComplex z0 = peekColour( iQ , 0 , 0 ) ;
  const LatticeComplex z1 = peekColour( iQ , 0 , 1 ) ; 
  const LatticeComplex Z = sqrt( z0*z0 + z1*adj(z1) ) ;
  const LatticeComplex f0 = cos( Z ) ;
  const LatticeComplex f1 = sin( Z )/Z ;
  e_iQ = 1.0 ;
  e_iQ = f0 * e_iQ + timesMinusI(f1) * iQ ;
#else
  const LatticeColourMatrix iQ2 = iQ*iQ ;
    
  // sign in c0 from the conventions on the Ta
  LatticeComplex u = -imag(trace(iQ2*iQ))*0.3333333333333333148 ;
  LatticeComplex w = -real(trace(iQ2))*0.5;
  LatticeComplex f0 = 0.3849001794597505244*w ;
  w = sqrt(w) ;
  f0 = f0*w ;
  f0 = acos(u/f0)*0.3333333333333333148;
  u = w*(0.5773502691896257311)*cos(f0);
  w = w*sin(f0);
    
  LatticeComplex f2 = timesI( func_xi0(w) );
  LatticeComplex u2 = u * u;
  LatticeComplex w2 = w * w;
  // set w to cos(w) as the actual value of w is not used after here
  w = cos(w);
    
  const LatticeComplex emiu = cos(u) - timesI(sin(u));
  //const LatticeComplex e2iu = adj(emiu)*adj(emiu) ;
  u *= 2. ;
  LatticeComplex e2iu = cos(u) + timesI(sin(u));

  f0 = e2iu * (u2 - w2) + emiu * ((8.0*u2 * w) + (u * (3.0*u2 + w2) * f2));
  LatticeComplex f1 = e2iu*u - emiu * ((u * w) - (3.0*u2 - w2) * f2);
  f2 = e2iu - emiu * (w + (1.5*u) * f2);

  w = 1.0 ; 
  w = w / (9.0 * u2 - w2);  // reals
  f0 = f0 * w ;
  f1 = f1 * w ;
  f2 = f2 * w ;
    
  e_iQ = 1.0 ;
  e_iQ = f0 * e_iQ + timesMinusI(f1) * iQ - f2 * iQ2;
#endif
}

static void
exponentiate_iQ_new2( LatticeColourMatrix& e_iQ,
		      const LatticeColourMatrix& iQ)
{
  e_iQ = 1.0 ;
  {
    autoView( e_iQ_v , e_iQ  , AcceleratorWrite ) ;      
    autoView( iQ_v   , iQ    , AcceleratorRead ) ;
    accelerator_for(ss,e_iQ_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	const auto iQ2 = iQ_v[ss]*iQ_v[ss] ;
	// sign in c0 from the conventions on the Ta
	auto u = -imag(trace(iQ2*iQ_v[ss]))*0.3333333333333333148 ;
	auto w = -real(trace(iQ2))*0.5;
	auto f0 = 0.3849001794597505244*w ;
	w = sqrt(w) ;
	f0 = f0*w ;
	f0 = acos(u/f0)*0.3333333333333333148;
	u = w*(0.5773502691896257311)*cos(f0);
	w = w*sin(f0);
	auto f2 = timesI( sin(w)/w );
	auto u2 = u * u;
	auto w2 = w * w;
	// set w to cos(w) as the actual value of w is not used after here
	w = cos(w);
	const auto emiu = cos(u) - timesI(sin(u));
	u = 2.*u ;
	auto e2iu = cos(u) + timesI(sin(u));
	f0 = e2iu * (u2 - w2) + emiu * ((8.0*u2 * w) + (u * (3.0*u2 + w2) * f2));
	auto f1 = e2iu*u - emiu * ((u * w) - (3.0*u2 - w2) * f2);
	f2 = e2iu - emiu * (w + (1.5*u) * f2);
	w = 1.0 ; 
	w = w / (9.0 * u2 - w2);  // reals
	f0 = f0 * w ;
	f1 = f1 * w ;
	f2 = f2 * w ;
	e_iQ_v[ss] = f0*e_iQ_v[ss] + timesMinusI(f1)*iQ_v[ss] - f2*iQ2;	
      });
  }
}

static void
set_iLambda_old( LatticeColourMatrix& iLambda,
		 LatticeColourMatrix& e_iQ,
		 const LatticeColourMatrix& iQ,
		 const LatticeColourMatrix& Sigmap,
		 const LatticeColourMatrix& GaugeK)
{
    GridBase* grid = iQ.Grid();
    LatticeColourMatrix iQ2(grid), iQ3(grid), B1(grid), B2(grid), USigmap(grid);
    LatticeColourMatrix unity(grid);
    unity = 1.0;

    LatticeComplex u(grid), w(grid);
    LatticeComplex f0(grid), f1(grid), f2(grid);
    LatticeComplex xi0(grid), xi1(grid), tmp(grid);
    LatticeComplex u2(grid), w2(grid), cosw(grid);
    LatticeComplex emiu(grid), e2iu(grid), qt(grid), fden(grid);
    LatticeComplex r01(grid), r11(grid), r21(grid), r02(grid), r12(grid);
    LatticeComplex r22(grid), tr1(grid), tr2(grid);
    LatticeComplex b10(grid), b11(grid), b12(grid), b20(grid), b21(grid),
      b22(grid);
    LatticeComplex LatticeUnitComplex(grid);

    LatticeUnitComplex = 1.0;

    // Exponential
    iQ2 = iQ * iQ;
    iQ3 = iQ * iQ2;
    set_uw_old(u, w, iQ2, iQ3);
    set_fj_old(f0, f1, f2, u, w);
    e_iQ = f0 * unity + timesMinusI(f1) * iQ - f2 * iQ2;

    // Getting B1, B2, Gamma and Lambda
    // simplify this part, reduntant calculations in set_fj
    xi0 = func_xi0(w);
    xi1 = func_xi1(w);
    u2 = u * u;
    w2 = w * w;
    cosw = cos(w);

    emiu = cos(u) - timesI(sin(u));
    e2iu = cos(2.0 * u) + timesI(sin(2.0 * u));

    r01 = (2.0 * u + timesI(2.0 * (u2 - w2))) * e2iu +
      emiu * ((16.0 * u * cosw + 2.0 * u * (3.0 * u2 + w2) * xi0) +
	      timesI(-8.0 * u2 * cosw + 2.0 * (9.0 * u2 + w2) * xi0));

    r11 = (2.0 * LatticeUnitComplex + timesI(4.0 * u)) * e2iu +
      emiu * ((-2.0 * cosw + (3.0 * u2 - w2) * xi0) +
	      timesI((2.0 * u * cosw + 6.0 * u * xi0)));

    r21 =
      2.0 * timesI(e2iu) + emiu * (-3.0 * u * xi0 + timesI(cosw - 3.0 * xi0));

    r02 = -2.0 * e2iu +
      emiu * (-8.0 * u2 * xi0 +
	      timesI(2.0 * u * (cosw + xi0 + 3.0 * u2 * xi1)));

    r12 = emiu * (2.0 * u * xi0 + timesI(-cosw - xi0 + 3.0 * u2 * xi1));

    r22 = emiu * (xi0 - timesI(3.0 * u * xi1));

    fden = LatticeUnitComplex / (2.0 * (9.0 * u2 - w2) * (9.0 * u2 - w2));

    b10 = 2.0 * u * r01 + (3.0 * u2 - w2) * r02 - (30.0 * u2 + 2.0 * w2) * f0;
    b11 = 2.0 * u * r11 + (3.0 * u2 - w2) * r12 - (30.0 * u2 + 2.0 * w2) * f1;
    b12 = 2.0 * u * r21 + (3.0 * u2 - w2) * r22 - (30.0 * u2 + 2.0 * w2) * f2;

    b20 = r01 - (3.0 * u) * r02 - (24.0 * u) * f0;
    b21 = r11 - (3.0 * u) * r12 - (24.0 * u) * f1;
    b22 = r21 - (3.0 * u) * r22 - (24.0 * u) * f2;

    b10 *= fden;
    b11 *= fden;
    b12 *= fden;
    b20 *= fden;
    b21 *= fden;
    b22 *= fden;

    B1 = b10 * unity + timesMinusI(b11) * iQ - b12 * iQ2;
    B2 = b20 * unity + timesMinusI(b21) * iQ - b22 * iQ2;
    USigmap = GaugeK * Sigmap;

    tr1 = trace(USigmap * B1);
    tr2 = trace(USigmap * B2);

    LatticeColourMatrix QUS = iQ * USigmap;
    LatticeColourMatrix USQ = USigmap * iQ;

    LatticeColourMatrix iGamma = tr1 * iQ - timesI(tr2) * iQ2 +
      timesI(f1) * USigmap + f2 * QUS + f2 * USQ;
    
    iLambda = Ta(iGamma);
}

// OK we just fully accelerator-thread this guy as it is costly
static void
set_iLambda_new( LatticeColourMatrix &iLambda,
		 LatticeColourMatrix &e_iQ,
		 const LatticeColourMatrix &iQ,
		 const LatticeColourMatrix &Sigmap,
		 const LatticeColourMatrix &GaugeK)
{
  GridBase* grid = iQ.Grid();
  LatticeComplex Id(grid); Id = 1.0;  
  e_iQ = GaugeK*Sigmap;
  iLambda = 1.0 ;
  {
    autoView( iLambda_v , iLambda , AcceleratorWrite ) ;
    autoView( e_iQ_v    , e_iQ    , AcceleratorWrite ) ;      
    autoView( iQ_v      , iQ      , AcceleratorRead ) ;
    autoView( Id_v      , Id      , AcceleratorRead ) ;
    accelerator_for(ss,iLambda_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	const auto iQ2 = iQ_v[ss]*iQ_v[ss] ;
	// sign in c0 from the conventions on the Ta
	auto u = -imag(trace(iQ2*iQ_v[ss]))*0.3333333333333333148 ;
	auto w = -real(trace(iQ2))*0.5;
	auto f0 = 0.3849001794597505244*w ;
	w = sqrt(w) ;
	f0 = f0*w ;
	f0 = acos(u/f0)*0.3333333333333333148;
	u = w*(0.5773502691896257311)*cos(f0);
	w = w*sin(f0);
	const auto xi0 = sin(w)/w; 
	auto f2        = timesI( xi0 );
	const auto u2  = u * u;
	const auto w2  = w * w;
	const auto xi1 = cos(w)/(w * w) - sin(w)/(w*w*w); //func_xi1(w);
	// set w to cos(w) as the actual value of w is not used after here
	w = cos(w);
	auto emiu = cos(u) - timesI(sin(u));
	//LatticeComplex e2iu = adj(emiu)*adj(emiu) ; weirdly sensitive to this ....
	u = 2.*u ;
	auto e2iu = cos(u) + timesI(sin(u));
	f0 = e2iu * (u2 - w2) + emiu * ((8.0*u2 * w) + (u * (3.0*u2 + w2) * f2));
	auto f1 = e2iu*u - emiu * ((u * w) - (3.0*u2 - w2) * f2);
	f2 = e2iu - emiu * (w + (1.5*u) * f2);
	const auto r01 = (u + timesI(2.0*(u2 - w2))) * e2iu +
	  emiu * ((8.*u*w + u * (3.0 * u2 + w2) * xi0) +
		  timesI(-8.0 * u2 * w + 2.0 * (9.0 * u2 + w2) * xi0));
	const auto r11 = (2.0 * Id_v[ss] + timesI(2.0*u)) * e2iu +
	  emiu * ((-2.0 * w + (3.0 * u2 - w2) * xi0) +
		  timesI(u*(w + 3.0*xi0)));
	const auto r21 = 2.0*timesI(e2iu) + emiu * (-1.5*u * xi0 + timesI(w - 3.0 * xi0));
	const auto r02 = -2.0 * e2iu +
	  emiu * (-8.0*u2*xi0 + timesI(u * (w + xi0 + 3.0*u2*xi1)));
	const auto r12 = emiu * (u * xi0 + timesI(-w - xi0 + 3.0*u2*xi1));
	const auto r22 = emiu * (xi0 - timesI(1.5*u * xi1));
	w = 1.0 ; 
	w = w / (9.0 * u2 - w2);  // reals
	f0 = f0 * w ;
	f1 = f1 * w ;
	f2 = f2 * w ;
	w *= (0.5*w) ;
	// B1
	auto EMIU = 3.*u2 - w2 ;
	auto E2IU = 30.*u2 + 2.*w2 ;
	auto b0_v = u*r01 + EMIU*r02 - E2IU*f0;
	auto b1_v = u*r11 + EMIU*r12 - E2IU*f1;
	auto b2_v = u*r21 + EMIU*r22 - E2IU*f2;
	b0_v *= w;
	b1_v *= w;
	b2_v *= w;
	auto B1 = b0_v*iLambda_v[ss] + timesMinusI(b1_v)*iQ_v[ss] - b2_v*iQ2;
	// B2
	EMIU = 1.5*u ;
	E2IU = 12.*u ;	  
	b0_v = r01 - EMIU*r02 - E2IU*f0;
	b1_v = r11 - EMIU*r12 - E2IU*f1;
	b2_v = r21 - EMIU*r22 - E2IU*f2;
	b0_v *= w;
	b1_v *= w;
	b2_v *= w;
	auto B2 = b0_v*iLambda_v[ss] + timesMinusI(b1_v)*iQ_v[ss] - b2_v*iQ2;
	// compute iLambda
	iLambda_v[ss]  = trace(e_iQ_v[ss] * B1) * iQ_v[ss] ;
	iLambda_v[ss] -= timesI( trace(e_iQ_v[ss] * B2) )*iQ2 ;
	iLambda_v[ss] += timesI(f1) * e_iQ_v[ss] ;
	iLambda_v[ss] += f2*( iQ_v[ss]*e_iQ_v[ss] + e_iQ_v[ss]*iQ_v[ss] ) ;
	iLambda_v[ss]  = Ta( iLambda_v[ss] ) ;
	// exponentiate
	B1 = 1 ;
	e_iQ_v[ss] = f0*B1 + timesMinusI(f1)*iQ_v[ss] - f2*iQ2 ;
      });
  }
}


static void
exponentiate_iQ_old( LatticeColourMatrix& e_iQ,
		     const LatticeColourMatrix& iQ)
{
  GridBase* grid = iQ.Grid();
  LatticeColourMatrix unity(grid);
  unity = 1.0;

  LatticeColourMatrix iQ2(grid), iQ3(grid);
  LatticeComplex u(grid), w(grid);
  LatticeComplex f0(grid), f1(grid), f2(grid);

  iQ2 = iQ * iQ;
  iQ3 = iQ * iQ2;

  //We should check sgn(c0) here already and then apply eq (34) from 0311018
  set_uw_old(u, w, iQ2, iQ3);
  set_fj_old(f0, f1, f2, u, w);

  e_iQ = f0 * unity + timesMinusI(f1) * iQ - f2 * iQ2;
}

static void
smear( LatticeGaugeField &u_smr , const LatticeGaugeField &U )
{
  GridBase *grid = U.Grid();
  LatticeColourMatrix Cup(grid), tmp_staple(grid) , tmp( grid ) ; 
  std::vector<LatticeColourMatrix> u(Nd, grid);
  for (int d = 0; d < Nd; d++) {
    u[d] = PeekIndex<LorentzIndex>(U, d);
  }
  for(int mu=0; mu<Nd; ++mu){
    Cup = Zero();
    for(int nu=0; nu<Nd; ++nu){
      if (nu != mu) {
	tmp = Cshift(u[nu],mu,1) ;
	tmp_staple  = u[nu]*Cshift(u[mu],nu,1)*adj(tmp) ;
	
	tmp_staple += Cshift(adj(u[nu])*u[mu]*tmp,nu,-1) ;
	Cup += adj(tmp_staple)*0.1;
      }
    }
    pokeLorentz(u_smr, adj(Cup), mu); 
  }
}

static void
smear_nuunroll( LatticeGaugeField &u_smr , const LatticeGaugeField &U )
{
  GridBase *grid = U.Grid();
  LatticeColourMatrix tmp_staple(grid) ;
  LatticeColourMatrix tmp_staple21(grid) , tmp_staple22( grid ) ,  tmp_staple23( grid ) ;
  LatticeColourMatrix tmp11( grid ) , tmp21( grid ) ;
  LatticeColourMatrix tmp12( grid ) , tmp22( grid ) ;
  LatticeColourMatrix tmp13( grid ) , tmp23( grid ) ; 
  std::vector<LatticeColourMatrix> u(Nd, grid);
  for (int d = 0; d < Nd; d++) {
    u[d] = PeekIndex<LorentzIndex>(U, d);
  }

  autoView( tmp_staple_v   , tmp_staple   , AcceleratorWrite ) ;
  autoView( tmp_staple21_v , tmp_staple21 , AcceleratorWrite ) ;
  autoView( tmp_staple22_v , tmp_staple22 , AcceleratorWrite ) ;
  autoView( tmp_staple23_v , tmp_staple23 , AcceleratorWrite ) ;

  const int mp[4][3] = { {1,2,3} , {0,2,3} , {0,1,3} , {0,1,2} } ;
  for(int mu=0; mu<Nd; ++mu){
    
    autoView( umu_v          , u[mu]        , AcceleratorRead ) ;

    tmp11 = Cshift( u[mp[mu][0]], mu , 1 ) ;
    tmp21 = Cshift( u[mu], mp[mu][0] , 1 ) ;

    autoView( tmp11_v       , tmp11        , AcceleratorRead ) ;
    autoView( tmp21_v       , tmp21        , AcceleratorRead ) ;
    autoView( unu1_v        , u[mp[mu][0]] , AcceleratorRead ) ;

    tmp12 = Cshift( u[mp[mu][1]], mu , 1 ) ;
    tmp22 = Cshift( u[mu], mp[mu][1] , 1 ) ;

    autoView( tmp12_v       , tmp12    , AcceleratorRead ) ;
    autoView( tmp22_v       , tmp22    , AcceleratorRead ) ;
    autoView( unu2_v        , u[mp[mu][1]]    , AcceleratorRead ) ;

    tmp13 = Cshift( u[mp[mu][2]], mu , 1 ) ;
    tmp23 = Cshift( u[mu], mp[mu][2] , 1 ) ;

    autoView( tmp13_v       , tmp13    , AcceleratorRead ) ;
    autoView( tmp23_v       , tmp23    , AcceleratorRead ) ;
    autoView( unu3_v        , u[mp[mu][2]]    , AcceleratorRead ) ;
    
    accelerator_for(ss,unu1_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	tmp_staple_v[ss]   = unu1_v[ss]*tmp21_v[ss]*adj(tmp11_v[ss]) ;
	tmp_staple21_v[ss] = adj(unu1_v[ss])*umu_v[ss]*(tmp11_v[ss]) ;
	
	tmp_staple_v[ss]  += unu2_v[ss]*tmp22_v[ss]*adj(tmp12_v[ss]) ;
	tmp_staple22_v[ss] = adj(unu2_v[ss])*umu_v[ss]*(tmp12_v[ss]) ;
	
	tmp_staple_v[ss]  += unu3_v[ss]*tmp23_v[ss]*adj(tmp13_v[ss]) ;
	tmp_staple23_v[ss] = adj(unu3_v[ss])*umu_v[ss]*(tmp13_v[ss]) ;
      }) ;
    tmp_staple += Cshift(tmp_staple21,mp[mu][0],-1) ;      
    tmp_staple += Cshift(tmp_staple22,mp[mu][1],-1) ;      
    tmp_staple += Cshift(tmp_staple23,mp[mu][2],-1) ;      
    
    pokeLorentz(u_smr, 0.1*tmp_staple, mu); 
    
    #if 0
    int nu = mp[mu][0] ;
    tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
    {
      autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
      autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
      autoView( tmp_v         , tmp         , AcceleratorRead ) ;
      autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
      autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
      autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
      accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	  tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	  tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	}) ;
    }
    tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
    Cup  = tmp_staple*0.1;

    // second nu dir
    nu = mp[mu][1] ;
    tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
    {
      autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
      autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
      autoView( tmp_v         , tmp         , AcceleratorRead ) ;
      autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
      autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
      autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
      accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	  tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	  tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	}) ;
    }
    tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
    Cup += tmp_staple*0.1;

    // final orthogonal direction
    nu = mp[mu][2] ;
    tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
    {
      autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
      autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
      autoView( tmp_v         , tmp         , AcceleratorRead ) ;
      autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
      autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
      autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
      accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	  tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	  tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	}) ;
    }
    tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
    Cup  += tmp_staple*0.1;
    
    pokeLorentz(u_smr, (Cup), mu);
    #endif
  }
}

static void
smear_nuunrollv2( LatticeGaugeField &u_smr , const LatticeGaugeField &U )
{
  GridBase *grid = U.Grid();
  LatticeColourMatrix tmp_staple1(grid) , tmp_staple21( grid ) ;
  LatticeColourMatrix tmp_staple22( grid ) , tmp_staple23( grid ) ;
  std::vector<LatticeColourMatrix> u(Nd*Nd, grid);
  for (int d = 0; d < Nd; d++) {
    u[d+Nd*d] = PeekIndex<LorentzIndex>(U, d);
    for( int nu = 0 ; nu < Nd ; nu++ ) {
      if( nu != d ) {
	u[nu+d*Nd] = Cshift( u[d+Nd*d] , nu , 1 ) ;
      }
    }
  }
  
  const int mp[4][3] = { {1,2,3} , {0,2,3} , {0,1,3} , {0,1,2} } ;
  for(int mu=0; mu<Nd; ++mu){

    autoView( tmp_staple1_v  , tmp_staple1  , AcceleratorWrite ) ;
    autoView( tmp_staple21_v , tmp_staple21 , AcceleratorWrite ) ;
    autoView( tmp_staple22_v , tmp_staple22 , AcceleratorWrite ) ;
    autoView( tmp_staple23_v , tmp_staple23 , AcceleratorWrite ) ;

    autoView( umu_v   , u[mu+Nd*mu]               , AcceleratorRead ) ;
    
    autoView( tmp11_v , u[mu+Nd*mp[mu][0]]        , AcceleratorRead ) ;
    autoView( tmp21_v , u[mp[mu][0]+Nd*mu]        , AcceleratorRead ) ;
    autoView( unu1_v  , u[mp[mu][0]+Nd*mp[mu][0]] , AcceleratorRead ) ;
    
    autoView( tmp12_v , u[mu+Nd*mp[mu][1] ]       , AcceleratorRead ) ;
    autoView( tmp22_v , u[mp[mu][1]+Nd*mu]        , AcceleratorRead ) ;
    autoView( unu2_v  , u[mp[mu][1]+Nd*mp[mu][1]] , AcceleratorRead ) ;
    
    autoView( tmp13_v , u[mu+Nd*mp[mu][2]]        , AcceleratorRead ) ;
    autoView( tmp23_v , u[mp[mu][2]+Nd*mu]        , AcceleratorRead ) ;
    autoView( unu3_v  , u[mp[mu][2]+Nd*mp[mu][2]] , AcceleratorRead ) ;
    
    accelerator_for(ss,unu1_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	tmp_staple1_v[ss]  = unu1_v[ss]*tmp21_v[ss]*adj(tmp11_v[ss]) ;
	tmp_staple21_v[ss] = adj(unu1_v[ss])*umu_v[ss]*(tmp11_v[ss]) ;
	
	tmp_staple1_v[ss] += unu2_v[ss]*tmp22_v[ss]*adj(tmp12_v[ss]) ;
	tmp_staple22_v[ss] = adj(unu2_v[ss])*umu_v[ss]*(tmp12_v[ss]) ;
	
	tmp_staple1_v[ss] += unu3_v[ss]*tmp23_v[ss]*adj(tmp13_v[ss]) ;
	tmp_staple23_v[ss] = adj(unu3_v[ss])*umu_v[ss]*(tmp13_v[ss]) ;
      }) ;
    // cshift downward cups
    tmp_staple1 += Cshift(tmp_staple21,mp[mu][0],-1) ;      
    tmp_staple1 += Cshift(tmp_staple22,mp[mu][1],-1) ;      
    tmp_staple1 += Cshift(tmp_staple23,mp[mu][2],-1) ;      
    
    pokeLorentz(u_smr, 0.1*tmp_staple1, mu); 
  }
}

template <class Gimpl>
static void derivative( LatticeGaugeField& SigmaTerm,
			const LatticeGaugeField& iLambda,
			const LatticeGaugeField& U)
{
  GridBase *grid = U.Grid();

  WilsonLoops<Gimpl> WL;
  LatticeColourMatrix staple(grid), u_tmp(grid);
  LatticeColourMatrix iLambda_mu(grid), iLambda_nu(grid);
  LatticeColourMatrix U_mu(grid), U_nu(grid);
  LatticeColourMatrix sh_field(grid), temp_Sigma(grid);
  Real rho_munu = 0.1 , rho_numu = 0.1 ;

  Real rho[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 } ;

      for(int mu = 0; mu < Nd; ++mu){
      U_mu       = peekLorentz(      U, mu);
      iLambda_mu = peekLorentz(iLambda, mu);

      for(int nu = 0; nu < Nd; ++nu){
	if(nu==mu) continue;
	U_nu       = peekLorentz(      U, nu);
	iLambda_nu = peekLorentz(iLambda, nu);

	rho_munu = rho[mu + Nd * nu];
	rho_numu = rho[nu + Nd * mu];

	WL.StapleUpper(staple, U, mu, nu);

	temp_Sigma = -rho_numu*staple*iLambda_nu;  //ok
	//-r_numu*U_nu(x+mu)*Udag_mu(x+nu)*Udag_nu(x)*Lambda_nu(x)
	Gimpl::AddLink(SigmaTerm, temp_Sigma, mu);

	sh_field = Cshift(iLambda_nu, mu, 1);// general also for Gparity?

	temp_Sigma = rho_numu*sh_field*staple; //ok
	//r_numu*Lambda_nu(mu)*U_nu(x+mu)*Udag_mu(x+nu)*Udag_nu(x)
	Gimpl::AddLink(SigmaTerm, temp_Sigma, mu);

	sh_field = Cshift(iLambda_mu, nu, 1);

	temp_Sigma = -rho_munu*staple*U_nu*sh_field*adj(U_nu); //ok
	//-r_munu*U_nu(x+mu)*Udag_mu(x+nu)*Lambda_mu(x+nu)*Udag_nu(x)
	Gimpl::AddLink(SigmaTerm, temp_Sigma, mu);
	
	staple = Zero();
	sh_field = Cshift(U_nu, mu, 1);

	temp_Sigma = -rho_munu*adj(sh_field)*adj(U_mu)*iLambda_mu*U_nu;
	temp_Sigma += rho_numu*adj(sh_field)*adj(U_mu)*iLambda_nu*U_nu;

	u_tmp = adj(U_nu)*iLambda_nu;
	sh_field = Cshift(u_tmp, mu, 1);
	temp_Sigma += -rho_numu*sh_field*adj(U_mu)*U_nu;
	sh_field = Cshift(temp_Sigma, nu, -1);
	Gimpl::AddLink(SigmaTerm, sh_field, mu);
      }
    }
}

//// New code 
template <class Gimpl>
static void derivative_new( LatticeGaugeField& SigmaTerm,
			    const LatticeGaugeField& iLambda,
			    const LatticeGaugeField& U)
{
  GridBase *grid = U.Grid();
  LatticeColourMatrix staple(grid), u_tmp(grid) ;
  LatticeColourMatrix sh_field(grid), temp_Sigma(grid) ;
  Real rho_munu = 0.1 , rho_numu = 0.1 ;
  Real rho[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 } ;

  std::vector<LatticeColourMatrix> u(Nd, grid), il(Nd, grid) ;
  for (int d = 0; d < Nd; d++) {
    u[d] = PeekIndex<LorentzIndex>(U, d);
    il[d] = PeekIndex<LorentzIndex>(iLambda, d);
  }

  for(int mu = 0; mu < Nd; ++mu){
    for(int nu = 0; nu < Nd; ++nu){
      if(nu==mu) continue;

      rho_munu = rho[mu + Nd * nu];
      rho_numu = rho[nu + Nd * mu]; 

      u_tmp = Cshift(u[nu],mu,1) ;
      staple = adj(u[nu]*Cshift(u[mu],nu,1)*adj(u_tmp)) ;
      sh_field = Cshift(il[nu], mu, 1);
      {
	autoView( tmp_v  , temp_Sigma , AcceleratorWrite ) ;
	autoView( sh_v   , sh_field , AcceleratorRead ) ;
	autoView( sh2_v  , Cshift(il[mu], nu, 1) , AcceleratorRead ) ;
	autoView( st_v   , staple   , AcceleratorRead ) ;
	autoView( unu_v  , u[nu]    , AcceleratorRead ) ;
	autoView( ilnu_v , il[nu]   , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	    tmp_v[ss] = -st_v[ss]*(rho_numu*ilnu_v[ss]+rho_munu*unu_v[ss]*sh2_v[ss]*adj(unu_v[ss]))
	      +rho_numu*sh_v[ss]*st_v[ss] ;
	  }) ;
      }
      Gimpl::AddLink(SigmaTerm, temp_Sigma, mu);

      // reset temp_Sigma
      staple = adj(u_tmp)*adj(u[mu]) ;
      // one can get a small speedup by pre-forming this product at the cost of a gauge field
      u_tmp = adj(u[nu])*il[nu];
      sh_field = Cshift(u_tmp, mu, 1);
      {
	autoView( tmp_v  , temp_Sigma , AcceleratorWrite ) ;
	autoView( sh_v   , sh_field , AcceleratorRead ) ;
	autoView( st_v   , staple   , AcceleratorRead ) ;
	autoView( unu_v  , u[nu]    , AcceleratorRead ) ;
	autoView( umu_v  , u[mu]    , AcceleratorRead ) ;
	autoView( ilmu_v , il[mu]   , AcceleratorRead ) ;
	autoView( ilnu_v , il[nu]   , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	    tmp_v[ss] = ( st_v[ss]*(-rho_munu*ilmu_v[ss]+rho_numu*ilnu_v[ss])
			  -rho_numu*sh_v[ss]*adj(umu_v[ss]) )*unu_v[ss] ;
	  }) ;
      }
      sh_field = Cshift(temp_Sigma, nu, -1);
      Gimpl::AddLink(SigmaTerm, sh_field, mu);
    }
  }
}

//// New code 
template <class Gimpl>
static void derivative_new2( LatticeGaugeField &SigmaTerm,
			     const LatticeGaugeField &iLambda,
			     const LatticeGaugeField &U)
{
  GridBase *grid = U.Grid();
  LatticeColourMatrix u_tmp1(grid) , u_tmp2(grid) , sh1(grid) , sh2(grid) ;
  LatticeColourMatrix temp_Sigma(grid) ;
  Real rho_munu = 0.1 , rho_numu = 0.1 ;
  Real rho[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 } ;

  std::vector<LatticeColourMatrix> u(Nd, grid), il(Nd, grid) ;
  for (int d = 0; d < Nd; d++) {
    u[d] = PeekIndex<LorentzIndex>(U, d);
    il[d] = PeekIndex<LorentzIndex>(iLambda, d);
  }

  for(int mu = 0; mu < Nd; ++mu){
    for(int nu = 0; nu < Nd; ++nu){
      if(nu==mu) continue;
      
      rho_munu = rho[mu + Nd * nu];
      rho_numu = rho[nu + Nd * mu]; 
      u_tmp1 = Cshift(u[nu]  , mu , 1 ) ;
      u_tmp2 = Cshift(u[mu]  , nu , 1 ) ;
      sh1    = Cshift(il[nu] , mu , 1 ) ;
      sh2    = Cshift(il[mu] , nu , 1 ) ;
      {
	autoView( temp_Sigma_v , temp_Sigma , AcceleratorWrite ) ;
	autoView( u_tmp1_v     , u_tmp1     , AcceleratorWrite ) ;
	autoView( sh1_v        , sh1        , AcceleratorRead ) ;
	autoView( sh2_v        , sh2        , AcceleratorRead ) ;
	autoView( u_tmp2_v     , u_tmp2     , AcceleratorRead ) ;
	autoView( unu_v        , u[nu]      , AcceleratorRead ) ;
	autoView( umu_v        , u[mu]      , AcceleratorRead ) ;
	autoView( ilmu_v       , il[mu]     , AcceleratorRead ) ;
	autoView( ilnu_v       , il[nu]     , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), LatticeColourMatrix::vector_object::Nsimd(),{
	    auto st = adj(unu_v[ss]*u_tmp2_v[ss]*adj(u_tmp1_v[ss])) ;
	    temp_Sigma_v[ss] = -st*(rho_numu*ilnu_v[ss]+rho_munu*unu_v[ss]*sh2_v[ss]*adj(unu_v[ss]))
	      +rho_numu*sh1_v[ss]*st ;
	    st = adj(u_tmp1_v[ss])*adj(umu_v[ss]) ;
	    auto sh3 = adj(u_tmp1_v[ss])*sh1_v[ss] ;
	    u_tmp1_v[ss] = ( st*(-rho_munu*ilmu_v[ss]+rho_numu*ilnu_v[ss])
			    -rho_numu*sh3*adj(umu_v[ss]) )*unu_v[ss] ;
	  }) ;
      }
      sh1 = Cshift(u_tmp1, nu, -1) + temp_Sigma ;
      Gimpl::AddLink(SigmaTerm, sh1, mu);
    }
  }
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian               Grid(latt_size,simd_layout,mpi_layout);

  std::vector<int> seeds({1,2,3,4});
  GridParallelRNG          pRNG(&Grid);  pRNG.SeedFixedIntegers(seeds);

  LatticeGaugeField Umu(&Grid);
  SU<Nc>::HotConfiguration(pRNG,Umu);
  std::vector<LatticeColourMatrix> U(4,&Grid);

  for(int mu=0;mu<Nd;mu++){
    U[mu] = PeekIndex<LorentzIndex>(Umu,mu);
  }
  std::cout<<"Nc -> "<<Nc<<std::endl ;
  printf( "%d\n" , Nc ) ;

  // test the projection
  LatticeColourMatrix tmp(U[0].Grid()) , tmp2(U[0].Grid()) , iLambda1(U[0].Grid()) , iLambda2(U[0].Grid()) ;
  LatticeComplex u1(U[0].Grid()) , w1(U[0].Grid()) ;
  LatticeComplex u2(U[0].Grid()) , w2(U[0].Grid()) ;

  LatticeComplex f10(U[0].Grid()) , f11(U[0].Grid()) , f12(U[0].Grid()) ;
  LatticeComplex f20(U[0].Grid()) , f21(U[0].Grid()) , f22(U[0].Grid()) ;
  for (int mu = 0; mu < Nd; mu++) {

    tmp = Ta( U[mu] ) ;

    double start = usecond();
    set_uw_old( u1, w1, tmp*tmp, tmp*tmp*tmp ) ;
    std::cout<<"olduw " << (usecond()-start)/1E3 << std::endl ;

    start = usecond();
    set_uw_new( u2, w2, tmp*tmp, tmp*tmp*tmp ) ;
    std::cout<<"newuw " << (usecond()-start)/1E3 << std::endl ;
    std::cout<<"Norm u "<< norm2(u2-u1)<<std::endl ;
    std::cout<<"Norm w "<< norm2(w2-w1)<<std::endl ;

    // set the fs
    start = usecond();
    set_fj_old( f10, f11, f12, u1, w1 ) ;
    std::cout<<"oldfj " << (usecond()-start)/1E3 << std::endl ;
    

    start = usecond();
    set_fj_new( f20, f21, f22, u2, w2 ) ;
    std::cout<<"newj " << (usecond()-start)/1E3 << std::endl ;

    std::cout<<"Norm f0 "<< norm2(f10-f20)<<std::endl ;
    std::cout<<"Norm f1 "<< norm2(f11-f21)<<std::endl ;
    std::cout<<"Norm f2 "<< norm2(f12-f22)<<std::endl ;

    std::cout<<"******************"<<std::endl ;
    start = usecond();
    exponentiate_iQ_old( tmp, Ta(U[mu]) ) ;
    std::cout<<"eiQ_old " << (usecond()-start)/1E3 << std::endl ;

    start = usecond();
    exponentiate_iQ_new2( tmp2, Ta(U[mu]) ) ;
    std::cout<<"eiQ_new " << (usecond()-start)/1E3 << std::endl ;
    std::cout<<"Norm eiQ "<< norm2(tmp2-tmp)<<std::endl<<std::endl ;

    start = usecond();
    set_iLambda_old( iLambda1, tmp, Ta(U[mu]) , U[mu] , U[mu] ) ;
    std::cout<<"iLambda_old " << (usecond()-start)/1E3 << std::endl ;

    start = usecond();
    set_iLambda_new( iLambda2, tmp, Ta(U[mu]) , U[mu] , U[mu] ) ;
    std::cout<<"iLambda_new " << (usecond()-start)/1E3 << std::endl ;

    std::cout<<"Norm iL "<< norm2(iLambda2-iLambda1)<<std::endl<<std::endl ;
  }

  LatticeGaugeField u_smr(&Grid) , u_smr2(&Grid) ;
  double start = usecond() ;
  smear( u_smr , Umu ) ;
  std::cout<<"Cup computation "<< (usecond()-start)/1e3 << std::endl ;

  start = usecond() ;
  smear_nuunroll( u_smr2 , Umu ) ;
  std::cout<<"Cup nuunroll "<< (usecond()-start)/1e3 << std::endl ;

  for(int mu = 0 ; mu < Nd ; mu++ ) {
    std::cout<<"Norm "<< norm2( PeekIndex<LorentzIndex>(u_smr,mu)
				- PeekIndex<LorentzIndex>(u_smr2,mu) ) << std::endl ;
      }


  LatticeGaugeField dU(&Grid) , dU2(&Grid) ;
  start = usecond() ;
  derivative<PeriodicGimplR>( dU , u_smr2 , Umu ) ;
  std::cout<<"Derivative timer " << (usecond()-start)/1e3 << std::endl ;

  start = usecond() ;
  derivative_new2<PeriodicGimplR>( dU2 , u_smr2 , Umu ) ;
  std::cout<<"Derivative new timer " << (usecond()-start)/1e3 << std::endl ;

  for(int mu = 0 ; mu < Nd ; mu++ ) {
    std::cout<<"Norm "<< norm2( PeekIndex<LorentzIndex>(dU,mu)
				- PeekIndex<LorentzIndex>(dU2,mu) ) << std::endl ;
      }
  
  
  Grid_finalize();
}
