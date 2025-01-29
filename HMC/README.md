## HMC runs

Here are a few details about HMC runs

StartingType Options are TepidStart CheckpointStart CheckpintStartReseed
For reseed you need a viable rng file even though it gets ignored

When starting a run only use TepidStart and make the TrajL small and don't do the metropolis for ~10 configs. It is important to turn off NoMetropolisUntil when doing actual runs though!
