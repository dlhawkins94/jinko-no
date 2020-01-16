{- -}

module Neural.Digits where

import Control.DeepSeq
import Control.Monad
import Numeric.LinearAlgebra
import System.Random

import Neural.FF
import Neural.LayerSpawners
import Neural.MNIST
                          
-- 1st layer fn is a dummy. I might replace with a dedicated dummy layer fn
spawnDigitClassifierNet :: IO (FFNetwork)
spawnDigitClassifierNet = spawnFFNetwork [(fullyConnectedLayer, (0,0), (28, 28), layerOpts),
                                          (convLayer, (28, 28), (5,5), layerOpts {
                                              layerDepth = 5,
                                              pooling = maxPooling,
                                              poolingBP = maxPoolingBP,
                                              poolSize = (2,2)
                                              }),
                                          (fullyConnectedLayer, (12, 12), (10, 1), layerOpts)]

trainDigitClassifierNet :: FFNetwork -> [((Matrix Double), Int)] -> [((Matrix Double), Int)] -> Int -> IO (FFNetwork, [(Matrix Double, Int)])
trainDigitClassifierNet net trainingSet testSet nEpochs = go net trainingSet 1
  where go net ts i | i <= nEpochs = do
                        (newNet, newTS) <- getStdRandom (\gen -> sgd net gen ts 9 16 0.1 5.0)
                        let nCorrect = (evaluate newNet testSet 5.0)
                        
                        putStrLn $ "Epoch " ++ (show i) ++ " of " ++ (show nEpochs) ++ ": "
                          ++ (show nCorrect) ++ "/" ++ (show $ length testSet)

                        {-
                        -- print out features
                        mapM_ (\layer ->
                                if (length $ weights layer) > 1
                                then (printMNISTImage $
                                      (\(w_1:w_) ->
                                        foldl (\w w_ -> w ||| (konst 0.0 (5,3) :: Matrix Double) ||| w_) w_1 w_)
                                      $ weights layer) >> putStrLn ""
                                else putStrLn "") $ layers net -}
                          
                        go newNet newTS $ i + 1
                        
                    | otherwise = return (net, trainingSet)

testDigitClassifierNet :: FFNetwork -> [((Matrix Double), Int)] -> IO ()
testDigitClassifierNet net testSet = do
  let tests = take 3 testSet
  putStrLn "Example tests: "
  printMNISTImage (foldl (\mat (input,expected) -> mat ||| input) ((\(x,y) -> x) (head tests)) (tail tests))
          
  putStrLn $ "Guesses: " ++
    (show $ map (\(input,expected) -> convertOutput $ ffOut net [input]) tests)
          
  putStrLn $ "Full evaluation: " ++ (show $ evaluate net testSet 1.0) ++ "/" ++ (show $ length testSet)

