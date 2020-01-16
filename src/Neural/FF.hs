{-# LANGUAGE DeriveGeneric, DeriveAnyClass, FlexibleContexts, FlexibleInstances, TypeSynonymInstances #-}

module Neural.FF where

import Control.Monad
import Control.Monad.State.Strict
import Control.Parallel.Strategies
import Data.List.Split
import qualified Data.Ix as I
import qualified Data.Vector as V
import GHC.Generics
import Numeric.LinearAlgebra
import System.Random

import Neural.Util
import Numeric.Extra

instance Show (a -> b) where show a = "function"

data FFLayerOptions = FFLayerOptions {
  -- # of slices
  layerDepth :: Int,

  -- z_j -> y_j
  layerActivationFn :: [Matrix Double] -> [Matrix Double],

  -- z_j -> yPrime_j
  layerActivationFnDeriv :: [Matrix Double] -> [Matrix Double],

  -- Pooling functions 
  pooling :: (Int,Int) -> [Matrix Double] -> [Matrix Double],
  poolingBP :: (Int,Int) -> [Matrix Double] -> [Matrix Double] -> [Matrix Double],
  poolSize :: (Int,Int)
  }

data FFLayer = FFLayer {
  weights :: [Matrix Double],
  biases :: [Matrix Double],

  -- x_j -> (w_j, b_j) -> z_j
  zFn :: [Matrix Double] -> ([Matrix Double], [Matrix Double]) -> [Matrix Double],
  yFn :: [Matrix Double] -> [Matrix Double],
  
  -- nabla_b_j -> x_j -> nabla_w_j
  nablaW :: [Matrix Double] -> [Matrix Double] -> [Matrix Double],

  -- delta_j -> x_j -> nabla_b_j
  nablaB :: [Matrix Double] -> [Matrix Double] -> [Matrix Double],
  
  -- nabla_b_j -> z_i-> w_j -> delta_i
  backprop :: [Matrix Double] -> [Matrix Double] -> [Matrix Double] -> [Matrix Double]
    } deriving (Show, Generic, NFData)

data FFNetwork = FFNetwork {
  layers :: V.Vector(FFLayer)
  } deriving (Show, Generic, NFData)


-- descs is a list of [(spawner, dims, options)]
spawnFFNetwork :: [((Int, Int) -> (Int, Int) -> FFLayerOptions -> FFLayerOptions -> IO (FFLayer),
                   (Int, Int), (Int, Int), FFLayerOptions)] -> IO (FFNetwork)
spawnFFNetwork descs = liftM (FFNetwork . V.fromList)
                       $ zipWithM (\(ls_j, dims_i, dims_j, opts_j) opts_i ->
                                    ls_j dims_i dims_j opts_i opts_j)
                       (tail descs) (map (\(_,_,_,opts_) -> opts_) descs)
                       
-- Parallel matrix multiplication; left operand is split into rows.
(<||#>) :: (Matrix Double) -> (Matrix Double) -> (Matrix Double)
x <||#> y = fromRows $ parMap rdeepseq (<# y) $ toRows x

-- Same as above, but right operand is split into columns
(<#||>) :: (Matrix Double) -> (Matrix Double) -> (Matrix Double)
x <#||> y = fromColumns $ parMap rdeepseq (x #>) $ toColumns y

-- Parallel dense matrix/vector product.
-- (m><n) #||> (n) -> (m)
(#||>) :: (Matrix Double) -> (Vector Double) -> (Vector Double)
x #||> y = fromList $ parMap rdeepseq (<.> y) $ toRows x

-- Idk what this product is called but is sure as hell is parallel
-- splits second operand into a list
-- I should really see if this is faster than <#||>
-- (m) <.||> (n) -> (m,n)
(<.||>) :: (Vector Double) -> (Vector Double) -> (Matrix Double)
x <.||> y = fromColumns $ parMap rdeepseq (\y_i -> scale y_i x) (toList y)

-- feeds forward, collecting layer activations.
-- note y, z are reversed.
-- net -> x_1 -> ([y_n, .. y_1, x_1], [z_n, .. z_1])
ff :: FFNetwork -> [Matrix Double] -> (V.Vector([Matrix Double]), V.Vector([Matrix Double]))
ff net x = V.foldl activator (V.singleton x, V.empty) $ layers net
  where activator (y, z) (FFLayer w b zf yf _ _ _) =
          (\(y_j, z_j) -> (y_j `V.cons` y, z_j `V.cons` z)) $ (\a -> (yf a, a)) $ zf (V.head y) (w, b)

-- ff that only returns final output of the net.
-- net -> x_1 -> y_n
ffOut :: FFNetwork -> [Matrix Double] -> [Matrix Double]
ffOut net x = V.head $ fst $ ff net x

-- feeds forward, then passes error backwards
bp :: FFNetwork -> [Matrix Double] -> [Matrix Double] -> [([Matrix Double], [Matrix Double])]
bp net expected x = V.foldl passBack [(nabla_w_n, nabla_b_n)] kit
  where passBack ((nabla_w_k, nabla_b_k) : nablas) (x_j, z_j, layer_k, layer_j) =
          (nabla_w_j, nabla_b_j) : (nabla_w_k, nabla_b_k) : nablas
          where nabla_w_j = (nablaW layer_j) nabla_b_j x_j
                nabla_b_j = (nablaB layer_j) delta_j z_j
                delta_j = (backprop layer_k) nabla_b_k z_j (weights layer_k)

        -- passBack sees these args:
        -- y[n-1 .. 1], z[n-1 .. 1], layers[n .. 2], layers[n-1 .. 1]
        kit = V.zip4 (V.tail $ V.tail y) (V.tail z) (V.reverse $ V.tail $ ls) (V.reverse $ V.init ls)
        nabla_w_n = (nablaW $ V.last $ ls) nabla_b_n (y V.! 1)
        nabla_b_n = zipWith (-) (y V.! 0) expected
        ls = layers net
        (y, z) = ff net x


                     
-- Error/cost functions.

{-
crossEntropyDelta :: [Matrix Double] -> [Matrix Double] -> [Matrix Double]
crossEntropyDelta = zipWith (-)

crossEntropyError :: [Matrix Double] -> [Matrix Double] -> [Double]
crossEntropyError x y = map (\(x,y) -> sumElements $ - ((y * (cmap log x)) + (1.0 - y) * (cmap log $ 1.0 - x)))
                        $ zip x y
-}

-- Convert an output value to an actual output of the net
-- i.e. '5' -> [0,0,0,0,0,1,0,...]
buildOutput :: Int -> Int -> [Matrix Double]
buildOutput i len = [(len><1) (map (\x -> if x == i then 1.0 else 0.0) [0..len - 1])]

-- Convert a net's output to an integer output value
-- uses max value of the net.
-- i.e. [0,0.5,0.25,-0.1,0.999999,...] -> '4'
convertOutput :: [Matrix Double] -> Int
convertOutput output = (\(i,j) -> i) $ maxIndex $ head output
        
-- For a given test set, return the number the net gets correct 
evaluate :: FFNetwork -> [((Matrix Double), Int)] -> Double -> Int
evaluate net testSet reg =
  sum $ parMap rdeepseq
  (\(input, expected) ->
    let output = ffOut net [input]
    in if convertOutput output == expected then 1 else 0) testSet

-- perform stochastic gradient descent on the net.
-- each sample data is a tuple of (input, expected output) where the output is an integer representing the desired result.
-- maxClassIndex is the max classification ID any output can produce
-- eta is the learning rate.
-- reg is the regularization parameter (for L2 reg)
sgd :: FFNetwork -> StdGen -> [((Matrix Double), Int)] -> Int -> Int -> Double -> Double -> ((FFNetwork, [((Matrix Double), Int)]), StdGen)
sgd net gen trainingSet maxClassIndex batchSize eta reg = ((newNet, newTrainingSet), newGen)
  where newNet = foldl applyBatch net $ chunksOf batchSize newTrainingSet
        applyBatch net batch =
          net { layers = 
                  V.zipWith (\layer (nabla_w, nabla_b) ->
                             layer {
                                -- apply the changes to each layer of the net.
                                weights = zipWith adjustWeights (weights layer) nabla_w,
                                biases = zipWith adjustBiases (biases layer) nabla_b
                                }) (layers net)
                  -- backpropagate each member of the batch, summing the resulting change in the net.
                  -- this will yield [(nw, nb) ...] for each layer of the net.
                  $ V.fromList
                  $ (\(batch1 : batches) ->
                      foldl (zipWith (\(nw,nb) (dnw,dnb) -> (zipWith (+) nw dnw, zipWith (+) nb dnb)))
                      batch1 batches)
                  $ parMap rdeepseq
                  (\(input, expected) ->
                    bp net (buildOutput expected $ maxClassIndex + 1) [input]) batch
              }

        -- hmatrix gets a little confused when (Num a, Matrix b) => a * b happens
        -- so use `scale` for now.             
        adjustWeights w nabla_w = (scale (1.0 - eta * reg / (fromIntegral $ length trainingSet)) w)
                                  - (scale (eta / (fromIntegral batchSize)) nabla_w)
        adjustBiases b nabla_b = b - (scale (eta / (fromIntegral batchSize)) nabla_b)

        -- cost = 0.5 * reg * l2_norm_sq * (fromIntegral batchSize) / (fromIntegral $ length trainingSet)
        -- l2_norm_sq = V.sum $ V.map (\l -> sum $ parMap rdeepseq (\w -> sumElements $ w * w) $ weights l) $ layers net
          
        (newTrainingSet, newGen) = shuffle' trainingSet gen
