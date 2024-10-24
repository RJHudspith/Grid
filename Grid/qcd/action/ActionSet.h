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

//////////////////////////////////
// Indexing of tuple types
//////////////////////////////////

typedef enum { LeapFrogIntegrator ,
	       MinimumNorm2Integrator ,
	       ForceGradientIntegrator ,
	       OMF2_QPQPQ,
	       OMF4Integrator ,
	       OMF4VIntegrator ,
	       OMF4_5PIntegrator,
	       OMF4_5VIntegrator } SupportedIntegrator ;

inline SupportedIntegrator
IntStringToEnum( const std::string str )
{
  if( str == "LeapFrog" )      return LeapFrogIntegrator ;
  if( str == "MinimumNorm2" )  return MinimumNorm2Integrator ;
  if( str == "ForceGradient" ) return ForceGradientIntegrator ;
  if( str == "OMF2_QPQPQ" )    return OMF2_QPQPQ ;
  if( str == "OMF4" )          return OMF4Integrator;
  if( str == "OMF4V" )         return OMF4VIntegrator;
  if( str == "OMF4_5P" )       return OMF4_5PIntegrator ;
  if( str == "OMF4_5V" )       return OMF4_5VIntegrator ;
  // shit the bed
  assert( false ) ;
  return MinimumNorm2Integrator ;
}

inline std::string
IntEnumToString( const SupportedIntegrator Integrator )
{
  std::string str = "" ;
  switch( Integrator ) {
  case LeapFrogIntegrator :      str="Leapfrog"      ; break ;
  case MinimumNorm2Integrator :  str="MinimumNorm2"  ; break ;
  case ForceGradientIntegrator : str="ForceGradient" ; break ;
  case OMF2_QPQPQ :              str="OMF2_QPQPQ"    ; break ;
  case OMF4Integrator :          str="OMF4"          ; break ;
  case OMF4VIntegrator :         str="OMF4V"         ; break ;
  case OMF4_5PIntegrator :       str="OMF4_5P"       ; break ;
  case OMF4_5VIntegrator :       str="OMF4_5V"       ; break ;
  }
  return str ;
}

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
		       const SupportedIntegrator _Integrator = MinimumNorm2Integrator ) : 
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
