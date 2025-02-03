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

static void
set_iLambda_new( LatticeColourMatrix& iLambda,
		 LatticeColourMatrix& e_iQ,
		 const LatticeColourMatrix& iQ,
		 const LatticeColourMatrix& Sigmap,
		 const LatticeColourMatrix& GaugeK)
{
    GridBase* grid = iQ.Grid();
    LatticeColourMatrix B1(grid), B2(grid) ;
    LatticeComplex LatticeUnitComplex(grid);
    LatticeUnitComplex = 1.0;

    LatticeColourMatrix iQ2 = iQ*iQ;

    // sign in c0 from the conventions on the Ta
    LatticeComplex u = -imag(trace(iQ2*iQ))*0.3333333333333333148 ;
    LatticeComplex w = -real(trace(iQ2))*0.5;
    LatticeComplex f0 = 0.3849001794597505244*w ;
    w = sqrt(w) ;
    f0 = f0*w ;
    f0 = acos(u/f0)*0.3333333333333333148;
    u = w*(0.5773502691896257311)*cos(f0);
    w = w*sin(f0);

    const LatticeComplex xi0 = func_xi0(w);
    LatticeComplex f2 = timesI( xi0 );
    const LatticeComplex u2 = u * u;
    const LatticeComplex w2 = w * w;
    const LatticeComplex xi1 = func_xi1(w);
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
    exponentiate_iQ_new( tmp2, Ta(U[mu]) ) ;
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


  Grid_finalize();
}
