import Control.DeepSeq
import qualified Data.Vector as V
import Numeric.LinearAlgebra
import System.Environment

import Neural.Digits
import Neural.FF
import Neural.MNIST

main :: IO ()
main = do
  maybeTrainingSet <- loadMNISTDataSet "./data/handwriting/train-images-idx3-ubyte" "./data/handwriting/train-labels-idx1-ubyte"
  maybeTestSet <- loadMNISTDataSet "./data/handwriting/t10k-images-idx3-ubyte" "./data/handwriting/t10k-labels-idx1-ubyte"
  
  case (maybeTrainingSet, maybeTestSet) of
    (Just trainingSet, Just testSet) ->
      (do let nEpochs = 60
          net <- spawnDigitClassifierNet
          putStrLn $ V.foldl (++) "Weights: " $ V.map (show . size . head . weights) $ layers net
          putStrLn $ V.foldl (++) "Biases: " $ V.map (show . size . head . biases) $ layers net
          -- note -- We're only seeing 1 (b,w) for nested layers. something to do w the pooling?
          (randn 28 28) >>= (\x -> let o = bp net [(konst 0.0 (10,1) :: Matrix Double)] [x]
                                  in o `deepseq` (putStrLn $ show $ ((\a -> (length a, size $ head a)) $ snd $ o !! 0)))
          putStrLn "worked"

          putStrLn $ "Training net for " ++ (show nEpochs) ++ " epochs:"
          (newNet, newTS) <- trainDigitClassifierNet net (trainingSet) (testSet) nEpochs
          return ())

          -- putStrLn "Testing net:"
          -- testDigitClassifierNet newNet testSet)
      
    (_,_) -> putStrLn "Failed to load images"
  
