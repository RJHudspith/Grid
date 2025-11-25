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
  
  // Default constructor. rho is constant in all directions, optionally except for orthogonal dimension
  // Will die if grid is actually passed as NULL in APE smearing routines
  Smear_Stout(double rho = 1.0, GridBase *grid = NULL , int orthogdim = -1)
    : OrthogDim{orthogdim},
      SmearRho{ rho3D(rho,orthogdim) },
      OwnedBase{ new Smear_APE<Gimpl>(rho,grid) },
      SmearBase{OwnedBase.get()} {
    assert(Nc<4 && "Stout smearing currently implemented only for Nc==2 or 3");
  }

  /*! Default constructor. rho is constant in all directions, optionally except for orthogonal dimension */
  Smear_Stout(double rho = 1.0, int orthogdim = -1)
  : OrthogDim{orthogdim}, SmearRho{ rho3D(rho,orthogdim) }, OwnedBase{ new Smear_APE<Gimpl>(SmearRho) }, SmearBase{OwnedBase.get()} {
    GRID_ASSERT(Nc == 3 && "Stout smearing currently implemented only for Nc==3");
>>>>>>> upstream/develop
  }

  ~Smear_Stout() {}  // delete SmearBase...

  void smear(GaugeField& u_smr, const GaugeField& U) const {
    GaugeLinkField tmp(U.Grid()), Umu(U.Grid());
    std::cout << GridLogDebug << "Stout smearing started\n";
    SmearBase->smear(u_smr, U);
    #pragma unroll
    for (int mu = 0; mu < Nd; mu++) {
      if( mu == OrthogDim ) continue ;
      // u_smr = exp(iQ_mu)*U_mu apart from Orthogdim
      Umu = peekLorentz(U, mu);
      tmp = peekLorentz(u_smr, mu);
      exponentiate_iQ(tmp, Ta( tmp * adj(Umu)) );
      pokeLorentz(u_smr, tmp * Umu, mu);
    }
    std::cout << GridLogDebug << "Stout smearing completed\n";
  };

  void derivative(GaugeField& SigmaTerm,
		  const GaugeField& iLambda,
                  const GaugeField& Gauge) const {
    SmearBase->derivative(SigmaTerm, iLambda, Gauge);
  };

  void BaseSmear(GaugeField& C, const GaugeField& U) const {
    SmearBase->smear(C, U);
  };

  void
  exponentiate_iQ( GaugeLinkField &e_iQ,
		   const GaugeLinkField &iQ) const {
       e_iQ = 1.0 ;
    {
      autoView( e_iQ_v , e_iQ  , AcceleratorWrite ) ;
      autoView( iQ_v   , iQ    , AcceleratorRead ) ;
      accelerator_for(ss,e_iQ_v.size(), GaugeLinkField::vector_object::Nsimd(),{
#if (Config_Nc == 2)
          auto z0 = peekColour( iQ_v[ss] , 0 , 0 ) ;
          auto z1 = peekColour( iQ_v[ss] , 0 , 1 ) ;
          auto Z  = sqrt( z0*z0 + z1*adj(z1) ) ;
          const auto f0 = cos( Z ) ;
          const auto f1 = sin( Z )/Z ;
          e_iQ = f0*e_iQ + timesMinusI(f1)*iQ_v[ss] ;
#else
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
#endif
        });
    }
  };
};

NAMESPACE_END(Grid);
