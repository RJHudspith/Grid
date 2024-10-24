    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_hmc_IwasakiGauge.cc

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>

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

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
#include <Grid/Grid.h>


int main(int argc, char **argv) {
  using namespace Grid;

  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  // here make a routine to print all the relevant information on the run
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

  SupportedIntegrator Inter[ 9 ] =
    {
      OMF2_3StepVIntegrator, OMF2_3StepPIntegrator,
      OMF2_5StepVIntegrator, OMF2_5StepPIntegrator, ForceGradientIntegrator,
      OMF4_9StepVIntegrator, OMF4_9StepPIntegrator,
      OMF4_11StepVIntegrator, OMF4_11StepPIntegrator,
    } ;

  for( int i = 0 ; i < 9 ; i++ ) {
    for(double t=1.0; t>1E-2;t*=0.5 ) {

      // Checkpointer definition
      CheckpointerParameters CPparams;  
      CPparams.config_prefix = "ckpoint_lat";
      CPparams.rng_prefix = "ckpoint_rng";
      CPparams.saveInterval = 800;
      CPparams.format = "IEEE64BIG";

      RNGModuleParameters RNGpar;
      RNGpar.serial_seeds = "1 2 3 4 5";
      RNGpar.parallel_seeds = "6 7 8 9 10";

      
      std::cout<<"Recording dH trajectory for step size :: "<<t<<" || Int -> "<<
	IntEnumToString( Inter[i] ) << std::endl ;

      // Typedefs to simplify notation
      typedef GenericHMCRunner<Integrator> HMCWrapper;  // Uses the default minimum norm
      HMCWrapper TheHMC;

      // Grid from the command line
      TheHMC.Resources.AddFourDimGrid("gauge");
      TheHMC.Resources.SetRNGSeeds(RNGpar);
      TheHMC.Resources.LoadNerscCheckpointer(CPparams);

      // Construct observables
      typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
      TheHMC.Resources.AddObservable<PlaqObs>();
    
      RealD beta = 4.375 ;
      SymanzikGaugeActionR Iaction(beta);
      ActionLevel<HMCWrapper::Field> Level1(1,Inter[i]);
      Level1.push_back(&Iaction);
      TheHMC.TheAction.push_back(Level1);
    
      // HMC parameters are serialisable
      TheHMC.Parameters.StartTrajectory = 100 ;
      TheHMC.Parameters.Trajectories = 100 ;
      TheHMC.Parameters.StartingType = "CheckpointStart" ;
      TheHMC.Parameters.MD.MDsteps = 1 ;
      TheHMC.Parameters.MD.trajL   = t ;
    
      TheHMC.Run();  // no smearing
    }
  }
  Grid_finalize();

} // main
