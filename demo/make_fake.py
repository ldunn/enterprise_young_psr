#!/usr/bin/env python

import libstempo
import libstempo.toasim as toasim

psr = libstempo.tempopulsar("J0835-4510_glitch2.par", "J0835-4510_glitch2.tim")
psr['GLF0D_4'].fit = False
psr['GLTD_4'].fit = False
psr['GLF0D_4'].val = 1.6e-6
psr['GLTD_4'].val = 15.8
psr.name = "FAKE"
toasim.make_ideal(psr)
toasim.make_ideal(psr)
toasim.make_ideal(psr)
toasim.add_efac(psr)
psr.savepar('fake_nored.par')
psr.savetim('fake_nored.tim')
toasim.add_rednoise(psr, 1e-9, 6, tspan = 10000, components=500)
psr['GLF0D_4'].val = 1e-10
psr['GLTD_4'].val = 1e-3
psr.fit()
psr.fit()
psr.fit()
psr['GLF0D_4'].fit = True
psr['GLTD_4'].fit = True
psr.savepar('fake.par')
psr.savetim('fake.tim')
