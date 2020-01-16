{-# LANGUAGE FlexibleContexts, TypeFamilies #-}

module Neural.LayerSpawners where

import Control.Monad
import Control.Parallel.Strategies
import Numeric.LinearAlgebra

import Neural.FF
import Neural.Util
import Numeric.Extra

maxPooling :: (Int, Int) -> [Matrix Double] -> [Matrix Double]
maxPooling (m_pool, n_pool) = map (fromLists . (map (map maxElement)) . (toBlocksEvery m_pool n_pool))

-- propagates the gradient through the pooling. 
-- the maximum element in each $1 block of $2 is set to its corresponding gradient, and all other elements are set to zero.
maxPoolingBP :: (Int, Int) -> [Matrix Double] -> [Matrix Double] -> [Matrix Double]
maxPoolingBP (m_pool, n_pool) = zipWith (\nb_k x_k -> fromBlocks $ zipWith (zipWith demax)
                                                     (toLists nb_k) (toBlocksEvery m_pool n_pool x_k))
   where demax nb__ x__ = accum (konst 0.0 (m_pool,n_pool) :: Matrix Double) (+) [(maxIndex x__, nb__)]

-- the default layer opts
layerOpts :: FFLayerOptions
layerOpts = FFLayerOptions {
  layerDepth = 1,
  layerActivationFn = map (cmap sigma),
  layerActivationFnDeriv = map (cmap sigmaPrime),
  pooling = (\_ -> id), -- default pooling and poolingBP don't affect input ∨ gradient
  poolingBP = (\_ nb _ -> nb),
  poolSize = (0,0)
  }

-- A fully connected layer is just one big slice that connects to all slices in the prev. layer -- there's only
-- one matrix in weights & biases each. For simplicity I'm using head to pull out these matrices.
-- all the matrices in the layer below are concatenated to fully connect the net . 
-- For layer j:
-- z_j(y_i) = (w_j . y_i) + b_j
-- y_j(z_j) = sigma(z_j(y_i))
-- ∇b_j = δ_j = ((tr w_k) . δ_k) * dy_j/dz_j
-- ∇w_j = δ_j . (tr y_i)

fullyConnectedLayer :: (Int, Int) -> (Int, Int) -> FFLayerOptions -> FFLayerOptions -> IO (FFLayer)
fullyConnectedLayer (m_i,n_i) (m_j,n_j) layerOpts_i layerOpts_j = do
  let depth_i = layerDepth layerOpts_i
      yFnDeriv_i = layerActivationFnDeriv layerOpts_i
      yFn_j = layerActivationFn layerOpts_j
  
  w <- randn (m_j * n_j) (m_i * n_i * depth_i)
  b <- randn (m_j * n_j) 1
  
  return (FFLayer {
             weights = [w],
             biases = [b],
   
             zFn = (\x_j (w_j, b_j) ->
                     [reshape n_j $ flatten
                      $ (head w_j <> (asColumn $ fullyFlatten x_j)) + (head b_j)]),
             yFn = yFn_j,
   
             nablaW = (\nabla_b_j x_j -> [(flatten $ head nabla_b_j) <.||> (fullyFlatten x_j)]),
             nablaB = (\a _ -> a),
             backprop = (\nabla_b_j z_i w_j ->
                          head $ toBlocksEvery m_i n_i
                          $ (reshape ((cols $ head z_i) * depth_i) $ tr (head w_j) #||> (flatten $ head nabla_b_j))
                          * (head $ yFnDeriv_i [fromBlocks [z_i]]))
             })

-- TODO: build pooling into the conv layer, then delete pooling layer. Don't be an ass

-- None of k conv slices in the same layer interact -- leaving out k subscript for convenience
-- let (d,d) be the size of the kernel window
-- z_i[x,y] = w_i.(x_i[x..x+d,y..y+d]) + b
-- y_i = sigma(z_i)
-- 
-- 
-- NOTES:
-- bias, weights are the same for each neuron in a slice
-- weights and bias form a kernel which is applied across input, the output of the kernel forming
--     a new matrix of size (input - d, input - d)

convLayer :: (Int, Int) -> (Int, Int) -> FFLayerOptions -> FFLayerOptions -> IO (FFLayer)
convLayer (m_i,n_i) (m_j,n_j) layerOpts_i layerOpts_j = do
  let depth_i = layerDepth layerOpts_i
      depth_j = layerDepth layerOpts_j
      yFnDeriv_i = layerActivationFnDeriv layerOpts_i
      yFn_j = layerActivationFn layerOpts_j
      pooling_j = pooling layerOpts_j
      poolingBP_j = poolingBP layerOpts_j
      (m_pool,n_pool) = poolSize layerOpts_j
      
  w <- replicateM depth_j $ randn m_j n_j
  b <- replicateM depth_j $ randn (m_i - m_j + 1) (n_i - n_j + 1)
  
  return (FFLayer {
             weights = w,
             biases = b,

             zFn = (\x_j (w_j, b_j) ->
                     let zf w_ b_ x_ = (corr2 w_ (fliprl . flipud $ x_)) + b_
                     in pooling_j (m_pool,n_pool)
                        $ zipWith (\w_ b_ -> (sum $ map (zf w_ b_) x_j) / (fromIntegral depth_j)) w_j b_j),
             yFn = yFn_j,

             nablaW = (\nabla_b_j x_j ->
                        let nwf nabla_b_ x_ = corr2 nabla_b_ x_
                        in if depth_i == 1
                           then parMap rdeepseq (\nabla_b_ -> nwf nabla_b_ (head x_j)) nabla_b_j
                           else parZipWith rdeepseq nwf nabla_b_j x_j),
             nablaB = poolingBP_j (m_pool,n_pool),
             backprop = (\nabla_b_j z_i w_j ->
                          zipWith3 (\nabla_b_ zp_ w_ -> (conv2 nabla_b_ w_) * zp_) nabla_b_j (yFnDeriv_i z_i) w_j)
             })
