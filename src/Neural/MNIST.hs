module Neural.MNIST where

import Control.Monad
import Data.Bits
import qualified Data.ByteString.Lazy as BSL
import Numeric.LinearAlgebra
import System.IO

readLittleEndian :: Handle -> IO (Int)
readLittleEndian hdl =
  (liftM BSL.reverse $ BSL.hGet hdl 4)
  >>= (\bytes -> return $
                foldl (\res (byte, i) -> res + (shift (fromIntegral byte) $ i * 8))
                0 $ zip (BSL.unpack bytes) [0..])


loadMNISTImages :: FilePath -> IO (Maybe [(Matrix Double)])
loadMNISTImages path =
  withBinaryFile path ReadMode readMNIST
  where readMNIST hdl =
          do test <- readLittleEndian hdl
             case test of
               2051 ->
                 (do imageCount <- readLittleEndian hdl
                     imageWidth <- readLittleEndian hdl
                     imageHeight <- readLittleEndian hdl

                     liftM Just $ readImages hdl imageWidth imageHeight)
                        
               _ -> return Nothing

        readImages hdl w h = do
          img <- (readImage hdl w h)
          case img of
            Just img -> liftM2 (:) (return img) (readImages hdl w h)
            Nothing -> return []

        readImage hdl w h =
          (BSL.hGet hdl (w * h))
          >>= (\bytes ->
                if (fromIntegral $ BSL.length bytes) == (fromIntegral $ (w * h))
                then return $ Just $
                     (w><h) $ map (\pixel -> fromIntegral pixel / 255.0) $ BSL.unpack bytes
                else return Nothing)

loadMNISTLabels :: FilePath -> IO (Maybe [Int])
loadMNISTLabels path =
  withBinaryFile path ReadMode readMNIST
  where readMNIST hdl =
          do test <- readLittleEndian hdl
             case test of
               2049 -> (readLittleEndian hdl) >>= (readLabels hdl)
               _ -> return Nothing

        readLabels hdl labelCount =
          (BSL.hGet hdl labelCount)
          >>= (\bytes ->
                if (fromIntegral $ BSL.length bytes) == (fromIntegral labelCount)
                then return $ Just $ map fromIntegral $ BSL.unpack bytes
                else return Nothing)

loadMNISTDataSet :: FilePath -> FilePath -> IO (Maybe [((Matrix Double), Int)])
loadMNISTDataSet imagesPath labelsPath =
  do maybeImages <- loadMNISTImages imagesPath
     maybeLabels <- loadMNISTLabels labelsPath

     case (maybeImages,maybeLabels) of
       (Just images, Just labels) -> return $ Just $ zip images labels
       (_,_) -> return Nothing
          
printMNISTImage :: (Matrix Double) -> IO ()
printMNISTImage image = putStr $ foldl (++) "" $ map printRow (toLists image)
  where printRow row = (reverse $ foldl (\str e -> (if e > 0.5 then '#' else ' ') : str) [] row) ++ "\n"
