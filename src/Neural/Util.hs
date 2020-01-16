module Neural.Util where

import Control.Monad
import Control.Monad.ST
import Control.Parallel.Strategies
import Data.Array.ST
import Data.STRef
import Numeric.LinearAlgebra
import System.Random

 -- shamelessly nicked from https://wiki.haskell.org/Random_shuffle

shuffle' :: [a] -> StdGen -> ([a],StdGen)
shuffle' xs gen = runST (do
        g <- newSTRef gen
        let randomRST lohi = do
              (a,s') <- liftM (randomR lohi) (readSTRef g)
              writeSTRef g s'
              return a
        ar <- newArray n xs
        xs' <- forM [1..n] $ \i -> do
                j <- randomRST (i,n)
                vi <- readArray ar i
                vj <- readArray ar j
                writeArray ar j vi
                return vj
        gen' <- readSTRef g
        return (xs',gen'))
  where
    n = length xs
    newArray :: Int -> [a] -> ST s (STArray s Int a)
    newArray n xs =  newListArray (1,n) xs

shuffleIO :: [a] -> IO [a]
shuffleIO xs = getStdRandom (shuffle' xs)

parZipWith :: Strategy c -> (a -> b -> c) -> [a] -> [b] -> [c]
parZipWith strat f x y = zipWith f x y `using` parList strat

parZipWith3 :: Strategy d -> (a -> b -> c -> d) -> [a] -> [b] -> [c] -> [d]
parZipWith3 strat f x y z = zipWith3 f x y z `using` parList strat

fullyFlatten :: [Matrix Double] -> Vector Double
fullyFlatten = vjoin . (parMap rdeepseq flatten)
