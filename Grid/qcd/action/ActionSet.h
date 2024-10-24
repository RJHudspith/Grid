/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/ActionSet.h

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>

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
#ifndef ACTION_SET_H
#define ACTION_SET_H

NAMESPACE_BEGIN(Grid);

// Ignore the 7step guys because they don't have the potential to be good
typedef enum { ForceGradientIntegrator ,
	       OMF2_3StepVIntegrator ,
	       OMF2_3StepPIntegrator ,
	       OMF2_5StepVIntegrator,
	       OMF2_5StepPIntegrator,
	       OMF4_9StepVIntegrator,
	       OMF4_9StepPIntegrator,
	       OMF4_11StepVIntegrator,
	       OMF4_11StepPIntegrator } SupportedIntegrator ;

inline SupportedIntegrator
IntStringToEnum( const std::string str )
{
  if( str == "ForceGradient" ) return ForceGradientIntegrator ;
  if( str == "OMF2_3StepV"  )  return OMF2_3StepVIntegrator ;
  if( str == "OMF2_3StepP"  )  return OMF2_3StepPIntegrator ;
  if( str == "OMF2_5StepV"  )  return OMF2_5StepVIntegrator ;
  if( str == "OMF2_5StepP"  )  return OMF2_5StepPIntegrator ;
  if( str == "OMF4_9StepV"  )  return OMF4_9StepVIntegrator ;
  if( str == "OMF4_9StepP"  )  return OMF4_9StepPIntegrator ;
  if( str == "OMF4_11StepV" )  return OMF4_11StepVIntegrator ;
  if( str == "OMF4_11StepP" )  return OMF4_11StepPIntegrator ;
  // shit the bed
  assert( false ) ;
  return OMF2_5StepVIntegrator ;
}

inline std::string
IntEnumToString( const SupportedIntegrator Integrator )
{
  std::string str = "" ;
  switch( Integrator ) {
  case ForceGradientIntegrator : str="ForceGradient" ; break ;
  case OMF2_3StepVIntegrator   : str="OMF2_3StepV"   ; break ;
  case OMF2_3StepPIntegrator   : str="OMF2_3StepP"   ; break ;
  case OMF2_5StepVIntegrator   : str="OMF2_5StepV"   ; break ;
  case OMF2_5StepPIntegrator   : str="OMF2_5StepP"   ; break ;
  case OMF4_9StepVIntegrator   : str="OMF4_9StepV"   ; break ;
  case OMF4_9StepPIntegrator   : str="OMF4_9StepP"   ; break ;
  case OMF4_11StepVIntegrator  : str="OMF4_11StepV"  ; break ;
  case OMF4_11StepPIntegrator  : str="OMF4_11StepP"  ; break ;
  }
  return str ;
}

// I know people will be frightened by change so I list possible synonyms from the
// literature so they recognise a familiar friend P = UpdaateP (Potential Force Term), Q = UpdateU (Kinetic)
inline std::string
IntEnumToString_Synonyms( const SupportedIntegrator Integrator )
{
  std::string str = "" ;
  switch( Integrator ) {
  case ForceGradientIntegrator : str="FGI_PQPQP"              ; break ;
  case OMF2_3StepVIntegrator   : str="Leapfrog PQP"           ; break ;
  case OMF2_3StepPIntegrator   : str="QPQ"                    ; break ;
  case OMF2_5StepVIntegrator   : str="MinimumNorm2 PQPQP MN2" ; break ;
  case OMF2_5StepPIntegrator   : str="QPQPQ"                  ; break ;
  case OMF4_9StepVIntegrator   : str=""                       ; break ;
  case OMF4_9StepPIntegrator   : str=""                       ; break ;
  case OMF4_11StepVIntegrator  : str="OMF4 (openQCD)"         ; break ;
  case OMF4_11StepPIntegrator  : str=""                       ; break ;
  }
  return str ;
}

//////////////////////////////////
// Indexing of tuple types
//////////////////////////////////
template <class T, class Tuple>
struct Index;

template <class T, class... Types>
struct Index<T, std::tuple<T, Types...>> {
  static const std::size_t value = 0;
};

template <class T, class U, class... Types>
struct Index<T, std::tuple<U, Types...>> {
  static const std::size_t value = 1 + Index<T, std::tuple<Types...>>::value;
};


////////////////////////////////////////////
// Action Level
// Action collection 
// in a integration level
// (for multilevel integration schemes)
////////////////////////////////////////////

template <class Field,
	  class Repr = NoHirep >
struct ActionLevel {
public:
  //Integrator Int ;
  
  unsigned int multiplier;

  // Fundamental repr actions separated because of the smearing
  typedef Action<Field>* ActPtr;

  // construct a tuple of vectors of the actions for the corresponding higher
  // representation fields
  typedef typename AccessTypes<Action, Repr>::VectorCollection action_collection;
  typedef typename AccessTypes<Action, Repr>::FieldTypeCollection action_hirep_types;

  action_collection actions_hirep;
  std::vector<ActPtr>& actions;
  SupportedIntegrator Integrator ;

  explicit ActionLevel(unsigned int mul = 1,
		       const SupportedIntegrator _Integrator = OMF2_5StepVIntegrator // this is the default MinimumNorm2 integrator
		       ) : 
    actions(std::get<0>(actions_hirep)), multiplier(mul),Integrator(_Integrator) {
    // initialize the hirep vectors to zero.
    // apply(this->resize, actions_hirep, 0); //need a working resize
    assert(mul >= 1);
  }

  template < class GenField >
  void push_back(Action<GenField>* ptr) {
    // insert only in the correct vector
    std::get< Index < GenField, action_hirep_types>::value >(actions_hirep).push_back(ptr);
  }

  template <class ActPtr>
  static void resize(ActPtr ap, unsigned int n) {
    ap->resize(n);
  }

  // Loop on tuple for a callable function
  template <std::size_t I = 1, typename Callable, typename ...Args>
  inline typename std::enable_if<I == std::tuple_size<action_collection>::value, void>::type apply(Callable, Repr& R,Args&...) const {}

  template <std::size_t I = 1, typename Callable, typename ...Args>
  inline typename std::enable_if<I < std::tuple_size<action_collection>::value, void>::type apply(Callable fn, Repr& R, Args&... arguments) const {
    fn(std::get<I>(actions_hirep), std::get<I>(R.rep), arguments...);
    apply<I + 1>(fn, R, arguments...);
  }  

};

// Define the ActionSet
template <class GaugeField, class R>
using ActionSet = std::vector<ActionLevel<GaugeField, R> >;

NAMESPACE_END(Grid);

#endif  // ACTION_SET_H
