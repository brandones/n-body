# N Body Simulator

If you're looking for a good example of how to make animations using
matplotlib's 3D scatterplots...

If you wanna see a real basic Newtonian time-stepped n-body model in
action...

If you've forgotten what the Earth and Venus's orbit around the sun
looks like...

Look no further.

### Our Solar System

`python sim_mpl.py home3`
Works. Pretty phresh. See it on (YouTube)[https://youtu.be/C2nn2NuIUGw]

### Alpha Centauri

Many, many brownie points to anyone who can figure out why the
Alpha Centauri simulation doesn't work. My only guess at the moment is
that it's because conservation of energy isn't respected, but I'm
skeptical since the orbit doesn't seem to get any better when the
timestep is reduced.

## A Note on Timestep Simulations

As mentioned above, in this simulation, energy and momentum are not
conserved. You may notice that this is not how physics works. This
is because e.g. energy is a smooth curve with particular points of
symmetry about which certain pairs of integrals are equal in real life,
while the same pairs of integrals on the time-stepped function, which
looks like any other step function, are unlikely to be.

