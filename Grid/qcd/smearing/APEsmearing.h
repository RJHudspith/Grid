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
  // Defines the gauge field types
  INHERIT_GIMPL_TYPES(Gimpl)
private:
  const std::vector<double> rho;/*!< Array of weights */

  //This member must be private - we do not want to control from outside
  std::vector<double> set_rho(const double common_rho) const {
    std::vector<double> res;
    for(int mn=0; mn<Nd*Nd; ++mn) res.push_back(common_rho);
    for(int mu=0; mu<Nd; ++mu) res[mu + mu*Nd] = 0.0;
    return res;
  }

  std::array<GaugeLinkField*,Nd> ust ;
  std::array<GaugeLinkField*,6> lst ;
  GaugeLinkField *u_tmp1 , *u_tmp2 ;
  GaugeLinkField *sh1 , *sh2 ;
  GaugeLinkField *temp_Sigma ;

  // alloc Umu                                                                                                         
  void
  setTmps( GridBase *grid ) {
    assert( grid != NULL ) ;
    std::cout<<"ALLOCATING SMEARED TEMP FIELDS *****"<<std::endl ;
    for( int mu = 0 ; mu < Nd ; mu++ ) {
      ust[mu] = new GaugeLinkField( grid ) ;
    }
    for( int mu = 0 ; mu < 6 ; mu++ ) {
      lst[mu] = new GaugeLinkField( grid ) ;
    }
    u_tmp1 = new GaugeLinkField( grid ) ;
    u_tmp2 = new GaugeLinkField( grid ) ;
    sh1    = new GaugeLinkField( grid ) ;
    sh2    = new GaugeLinkField( grid ) ;
    temp_Sigma = new GaugeLinkField( grid ) ;
  }

  void
  delTmps( ) const {
    std::cout<<"DESTROYING SMEARED TEMP FIELDS *****"<<std::endl ;
    for( int mu = 0 ; mu < Nd ; mu++ ) {
      if( ust[mu] != NULL ) delete ust[mu] ;
    }
    for( int mu = 0 ; mu < 6 ; mu++ ) {
      if( lst[mu] != NULL ) delete lst[mu] ;
    }
    if( u_tmp1 != NULL ) delete u_tmp1 ;
    if( u_tmp2 != NULL ) delete u_tmp2 ;
    if( sh1 != NULL    ) delete sh1 ;
    if( sh2 != NULL    ) delete sh2 ;
    if( temp_Sigma != NULL ) delete temp_Sigma ;
  }

public:

  // Constructors and destructors
  Smear_APE(const std::vector<double>& rho_):rho(rho_){} // check vector size
  Smear_APE(double rho_val):rho(set_rho(rho_val)){}
  Smear_APE(double rho_val, GridBase *grid ):rho(set_rho(rho_val)){
    setTmps(grid) ;
  }
  Smear_APE():rho(set_rho(1.0)){}
  ~Smear_APE(){
    delTmps() ;
  }

  ///////////////////////////////////////////////////////////////////////////////
  void smear(GaugeField& u_smr, const GaugeField& U)const{
    // faster version but doesn't work with weird boundaries!!! Use at your own risk!!
    assert( Nd==4 ) ;
    #pragma unroll
    for (int d = 0; d < Nd; d++) {
      *ust[d] = PeekIndex<LorentzIndex>(U, d);
    }
    autoView( tmp_staple1_v , (*u_tmp1) , AcceleratorWrite ) ;
    autoView( tmp_staple2_v , (*u_tmp2) , AcceleratorWrite ) ;
    const int mp[4][3] = { {1,2,3} , {0,2,3} , {0,1,3} , {0,1,2} } ;
    for( int mu = 0 ; mu < Nd ; mu++ ) {
      autoView( umu_v , (*ust[mu]) , AcceleratorRead ) ;
      int nu = mp[mu][0] ;
      Real rhomu = rho[mu+Nd*nu] ;
      *sh1 = Cshift(*ust[nu],mu,1) ; *sh2 = Cshift(*ust[mu],nu,1) ;
      {
        autoView( tmp_v  , (*sh1)     , AcceleratorRead ) ;
        autoView( tmp2_v , (*sh2)     , AcceleratorRead ) ;
        autoView( unu_v  , (*ust[nu]) , AcceleratorRead ) ;
        accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
            tmp_staple1_v[ss] = rhomu*(unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss]) );
            tmp_staple2_v[ss] = rhomu*(adj(unu_v[ss])*umu_v[ss]*tmp_v[ss]) ;
          }) ;
      }
      *u_tmp1 += Cshift(*u_tmp2,nu,-1) ;
      // second orthodir                                                                                            
      nu = mp[mu][1] ;
      rhomu = rho[mu+Nd*nu] ;
      *sh1 = Cshift(*ust[nu],mu,1) ; *sh2 = Cshift(*ust[mu],nu,1) ;
      {
        autoView( tmp_v         , (*sh1)     , AcceleratorRead ) ;
        autoView( tmp2_v        , (*sh2)     , AcceleratorRead ) ;
        autoView( unu_v         , (*ust[nu]) , AcceleratorRead ) ;
        accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
            tmp_staple1_v[ss] += rhomu*(unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss])) ;
            tmp_staple2_v[ss]  = rhomu*(adj(unu_v[ss])*umu_v[ss]*tmp_v[ss]) ;
          }) ;
      }
      *u_tmp1 += Cshift(*u_tmp2,nu,-1) ;
      // third othodir                                                                                              
      nu = mp[mu][2] ;
      rhomu = rho[mu+Nd*nu] ;
      *sh1 = Cshift(*ust[nu],mu,1) ; *sh2 = Cshift(*ust[mu],nu,1) ;
      {
        autoView( tmp_v         , (*sh1)     , AcceleratorRead ) ;
        autoView( tmp2_v        , (*sh2)     , AcceleratorRead ) ;
        autoView( unu_v         , (*ust[nu]) , AcceleratorRead ) ;
        accelerator_for(ss,unu_v.size(), GaugeField::vector_object::Nsimd(),{
            tmp_staple1_v[ss] += rhomu*(unu_v[ss]*tmp2_v[ss]*adj(tmp_v[ss])) ;
            tmp_staple2_v[ss]  = rhomu*(adj(unu_v[ss])*umu_v[ss]*tmp_v[ss]) ;
          }) ;
      }
      *u_tmp1 += Cshift(*u_tmp2,nu,-1) ;
      pokeLorentz(u_smr, *u_tmp1, mu);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  void derivative(GaugeField& SigmaTerm,
		  const GaugeField& iLambda,
		  const GaugeField& U)const{
    Real rho_munu = 0. , rho_numu = 0. ;
    #pragma unroll
    for (int d = 0; d < Nd; d++) {
      *ust[d] = PeekIndex<LorentzIndex>(U, d);
      *lst[d] = PeekIndex<LorentzIndex>(iLambda, d);
    }
    for(int mu = 0; mu < Nd; ++mu){
      autoView( umu_v        , (*ust[mu])    , AcceleratorRead ) ;
      autoView( ilmu_v       , (*lst[mu])    , AcceleratorRead ) ;
      #pragma unroll
      for(int nu = 0; nu < Nd; ++nu){
        if(nu==mu) continue;
        rho_munu = rho[mu + Nd * nu];
        rho_numu = rho[nu + Nd * mu];
        *u_tmp1  = Cshift(*ust[nu] , mu , 1 ) ;
        *u_tmp2  = Cshift(*ust[mu] , nu , 1 ) ;
        *sh1     = Cshift(*lst[nu] , mu , 1 ) ;
        *sh2     = Cshift(*lst[mu] , nu , 1 ) ;
        autoView( temp_Sigma_v , (*temp_Sigma) , AcceleratorWrite ) ;
        autoView( u_tmp1_v     , (*u_tmp1)     , AcceleratorWrite ) ;
        autoView( sh1_v        , (*sh1)        , AcceleratorRead ) ;
        autoView( sh2_v        , (*sh2)        , AcceleratorRead ) ;
        autoView( u_tmp2_v     , (*u_tmp2)     , AcceleratorRead ) ;
        autoView( unu_v        , (*ust[nu])    , AcceleratorRead ) ;
        autoView( ilnu_v       , (*lst[nu])    , AcceleratorRead ) ;
        accelerator_for(ss,unu_v.size(), GaugeLinkField::vector_object::Nsimd(),{
            auto st = adj(unu_v[ss]*u_tmp2_v[ss]*adj(u_tmp1_v[ss])) ;
            const auto tmp1 = unu_v[ss]*sh2_v[ss]*adj(unu_v[ss]) ;
            const auto tmp3 = rho_numu*ilnu_v[ss] ;
            temp_Sigma_v[ss] = -st*(tmp3+rho_munu*tmp1)+rho_numu*(sh1_v[ss]*st) ;
            st  = adj(umu_v[ss]*u_tmp1_v[ss]) ;
            const auto sh3 = adj(u_tmp1_v[ss])*sh1_v[ss] ;
            st  = st*(-rho_munu*ilmu_v[ss]+tmp3) ;
            st -= rho_numu*sh3*adj(umu_v[ss]) ;
            u_tmp1_v[ss] = st*unu_v[ss] ;
          }) ;
        *sh1 = Cshift( *u_tmp1, nu, -1) + *temp_Sigma ;
        Gimpl::AddLink(SigmaTerm, *sh1, mu);
      }
    }
  }
};

NAMESPACE_END(Grid);

