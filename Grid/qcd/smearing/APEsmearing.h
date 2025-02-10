/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/modules/plaquette.h

Copyright (C) 2017

Author: Guido Cossu <guido.cossu@ed.ac.uk>

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
			   /*  END LEGAL */
			   /*!
			     @brief Declaration of Smear_APE class for APE smearing
			   */

#pragma once

NAMESPACE_BEGIN(Grid);

/*!  @brief APE type smearing of link variables. */
template <class Gimpl>
class Smear_APE: public Smear<Gimpl>{
private:
  const std::vector<double> rho;/*!< Array of weights */

  //This member must be private - we do not want to control from outside
  std::vector<double> set_rho(const double common_rho) const {
    std::vector<double> res;

    for(int mn=0; mn<Nd*Nd; ++mn) res.push_back(common_rho);
    for(int mu=0; mu<Nd; ++mu) res[mu + mu*Nd] = 0.0;
    return res;
  }

public:
  // Defines the gauge field types
  INHERIT_GIMPL_TYPES(Gimpl)


  // Constructors and destructors
  Smear_APE(const std::vector<double>& rho_):rho(rho_){} // check vector size
  Smear_APE(double rho_val):rho(set_rho(rho_val)){}
  Smear_APE():rho(set_rho(1.0)){}
  ~Smear_APE(){}

  ///////////////////////////////////////////////////////////////////////////////
  void smear(GaugeField& u_smr, const GaugeField& U)const{
    // faster version but doesn't work with weird boundaries!!! Use at your own risk!!
    GridBase *grid = U.Grid();
    GaugeLinkField Cup(grid), tmp_staple(grid) , tmp_staple2(grid) , tmp( grid ), tmp2( grid ) ;
    std::vector<GaugeLinkField> u(Nd, grid);
    for (int d = 0; d < Nd; d++) {
      u[d] = PeekIndex<LorentzIndex>(U, d);
    }    
    const int mp[4][3] = { {1,2,3} , {0,2,3} , {0,1,3} , {0,1,2} } ;
    for(int mu=0; mu<Nd; ++mu){
      // first ortho dir
      int nu = mp[mu][0] ;
      Real rhomu = rho[mu+Nd*mp[mu][0]] ;
      tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
      {
	autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
	autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
	autoView( tmp_v         , tmp         , AcceleratorRead ) ;
	autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
	autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
	autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
	    tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	    tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	  }) ;
      }
      tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
      Cup  = tmp_staple*rhomu;

      // second nu dir
      nu = mp[mu][1] ;
      rhomu = rho[mu+Nd*mp[mu][1]] ;
      tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
      {
	autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
	autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
	autoView( tmp_v         , tmp         , AcceleratorRead ) ;
	autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
	autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
	autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
	    tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	    tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	  }) ;
      }
      tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
      Cup += tmp_staple*rhomu;

      // final ortho dir
      nu = mp[mu][2] ;
      rhomu = rho[mu+Nd*mp[mu][2]] ;
      tmp = Cshift(u[nu],mu,1) ; tmp2 = Cshift(u[mu],nu,1) ;
      {
	autoView( tmp_staple_v  , tmp_staple  , AcceleratorWrite ) ;
	autoView( tmp_staple2_v , tmp_staple2 , AcceleratorWrite ) ;
	autoView( tmp_v         , tmp         , AcceleratorRead ) ;
	autoView( tmp2_v        , tmp2        , AcceleratorRead ) ;
	autoView( unu_v         , u[nu]       , AcceleratorRead ) ;
	autoView( umu_v         , u[mu]       , AcceleratorRead ) ;
	accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
	    tmp_staple_v[ss] = unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) ;
	    tmp_staple2_v[ss] = adj(unu_v[ss])*umu_v[ss]*(tmp_v[ss]) ;
	  }) ;
      }
      tmp_staple += Cshift(tmp_staple2,nu,-1) ;      
      Cup  += tmp_staple*rhomu;
    
      pokeLorentz(u_smr, Cup, mu); 
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void derivative(GaugeField& SigmaTerm,
		  const GaugeField& iLambda,
		  const GaugeField& U)const{
    GridBase *grid = U.Grid();
    GaugeLinkField u_tmp1(grid) , u_tmp2(grid) , sh1(grid) , sh2(grid) , temp_Sigma(grid) ;
    std::vector<GaugeLinkField> u(Nd, grid), il(Nd, grid) ;
    Real rho_munu = 0. , rho_numu = 0. ;
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
          accelerator_for(ss,unu_v.size(), GaugeLinkField::vector_object::Nsimd(),{
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
};

NAMESPACE_END(Grid);

