module Numeric.Extra where

sigma :: Double -> Double
sigma x = 1.0 / (1.0 + (exp $ -x))

sigmaPrime :: Double -> Double
sigmaPrime x = (sigma x) * (1.0 - (sigma x))
