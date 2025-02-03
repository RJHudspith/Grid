/*************************************************************************************
 
 Grid physics library, www.github.com/paboyle/Grid
 
 Source file: ./lib/qcd/smearing/StoutSmearing.h
 
 Copyright (C) 2019
 
 Author: unknown
 Author: Felix Erben <ferben@ed.ac.uk>
 Author: Michael Marshall <Michael.Marshall@ed.ac.uk>

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License along
 with this program; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 
 See the full license in the file "LICENSE" in the top level distribution
 directory
 *************************************************************************************/
/*
  @file StoutSmearing.h
  @brief Declares Stout smearing class
*/
#pragma once

NAMESPACE_BEGIN(Grid);

/*!  @brief Stout smearing of link variable. */
template <class Gimpl>
class Smear_Stout : public Smear<Gimpl> {
 private:
  int OrthogDim = -1;
public:
  const std::vector<double> SmearRho;
private:
  // Smear<Gimpl>* ownership semantics:
  //    Smear<Gimpl>* passed in to constructor are owned by caller, so we don't delete them here
  //    Smear<Gimpl>* created within constructor need to be deleted as part of the destructor
  const std::unique_ptr<Smear<Gimpl>> OwnedBase; // deleted at destruction
  const Smear<Gimpl>* SmearBase; // Not owned by this object, so not deleted at destruction

  // only anticipated to be used from default constructor
  inline static std::vector<double> rho3D(double rho, int orthogdim){
    std::vector<double> rho3d(Nd*Nd);
    for (int mu=0; mu<Nd; mu++)
      for (int nu=0; nu<Nd; nu++)
        rho3d[mu + Nd * nu] = (mu == nu || mu == orthogdim || nu == orthogdim) ? 0.0 : rho;
    return rho3d;
  };
  
public:
  INHERIT_GIMPL_TYPES(Gimpl)

  /*! Stout smearing with base explicitly specified */
  Smear_Stout(Smear<Gimpl>* base) : SmearBase{base} {
    assert(Nc<4 && "Stout smearing currently implemented only for Nc==2 or 3");
  }

  /*! Construct stout smearing object from explicitly specified rho matrix */
  Smear_Stout(const std::vector<double>& rho_)
    : OwnedBase{new Smear_APE<Gimpl>(rho_)}, SmearBase{OwnedBase.get()} {
    std::cout << GridLogDebug << "Stout smearing constructor : Smear_Stout(const std::vector<double>& " << rho_ << " )" << std::endl;
    assert(Nc<4 && "Stout smearing currently implemented only for Nc== 2 or 3");
    }

  /*! Default constructor. rho is constant in all directions, optionally except for orthogonal dimension */
  Smear_Stout(double rho = 1.0, int orthogdim = -1)
  : OrthogDim{orthogdim}, SmearRho{ rho3D(rho,orthogdim) }, OwnedBase{ new Smear_APE<Gimpl>(SmearRho) }, SmearBase{OwnedBase.get()} {
    assert(Nc<4 && "Stout smearing currently implemented only for Nc==2 or 3");
  }

  ~Smear_Stout() {}  // delete SmearBase...

  void smear(GaugeField& u_smr, const GaugeField& U) const {
    GaugeField C(U.Grid());
    GaugeLinkField tmp(U.Grid()), Umu(U.Grid());

    std::cout << GridLogDebug << "Stout smearing started\n";

    // C contains the staples multiplied by some rho
    u_smr = U ; // set the smeared field to the current gauge field
    SmearBase->smear(C, U);

    for (int mu = 0; mu < Nd; mu++) {
      if( mu == OrthogDim ) continue ;
      // u_smr = exp(iQ_mu)*U_mu apart from Orthogdim
      Umu = peekLorentz(U, mu);
      tmp = peekLorentz(C, mu);
      exponentiate_iQ(tmp, Ta( tmp * adj(Umu)) );
      pokeLorentz(u_smr, tmp * Umu, mu);
    }
    std::cout << GridLogDebug << "Stout smearing completed\n";
  };

  void derivative(GaugeField& SigmaTerm, const GaugeField& iLambda,
                  const GaugeField& Gauge) const {
    SmearBase->derivative(SigmaTerm, iLambda, Gauge);
  };

  void BaseSmear(GaugeField& C, const GaugeField& U) const {
    SmearBase->smear(C, U);
  };

  void
  exponentiate_iQ( GaugeLinkField &e_iQ,
		   const GaugeLinkField &iQ) const {
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
  };

  void
  set_uw( LatticeComplex& u,
	  LatticeComplex& w,
	  const GaugeLinkField& iQ2,
	  const GaugeLinkField& iQ3) const {
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

  void
  set_fj( LatticeComplex& f0,
	  LatticeComplex& f1,
	  LatticeComplex& f2,
	  const LatticeComplex& u,
	  const LatticeComplex& w) const {
    const LatticeComplex u2 = u * u;
    const LatticeComplex w2 = w * w;
    const LatticeComplex cosw = cos(w);
    const LatticeComplex emiu = cos(u) - timesI(sin(u));
    //const LatticeComplex e2iu = adj(emiu)*adj(emiu) ;
    const LatticeComplex e2iu = cos(2.*u) + timesI(sin(2.*u)) ;
    LatticeComplex ixi0 = timesI( func_xi0(w) );  
    f0 = e2iu * (u2 - w2) + emiu * ((8.0 * u2 * cosw) + (2.*u*(3.0 * u2 + w2) * ixi0));
    f1 = e2iu * (2.0 * u) - emiu * ((2.0 * u * cosw) - (3.0 * u2 - w2) * ixi0);
    f2 = e2iu - emiu * (cosw + (3.0 * u) * ixi0);
    ixi0 = 1.0 ; 
    ixi0 = ixi0 / (9.0 * u2 - w2);
    f0 = f0 * ixi0 ;
    f1 = f1 * ixi0 ;
    f2 = f2 * ixi0 ;
  }

  LatticeComplex func_xi0(const LatticeComplex& w) const {
    // Definition from arxiv 0311018
    //if (abs(w) < 0.05) {w2 = w*w; return 1.0 - w2/6.0 * (1.0-w2/20.0 * (1.0-w2/42.0));}
    return sin(w) / w;
  }

  LatticeComplex func_xi1(const LatticeComplex& w) const {
    // Define a function to do the check
    // if( w < 1e-4 ) std::cout << GridLogWarning << "[Smear_stout] w too small:
    // "<< w <<"\n";
    return cos(w) / (w * w) - sin(w) / (w * w * w);
  }
};

NAMESPACE_END(Grid);
