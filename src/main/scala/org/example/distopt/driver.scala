package org.example.distopt

import org.apache.spark.{SparkConf, SparkContext}
import org.example.distopt.solvers._
import org.example.distopt.utils._
//import org.apache.spark.mllib.linalg.DenseVector
import breeze.linalg.DenseVector

object driver {

  def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[4]")
    val dataFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits","1").toInt
    val chkptDir = options.getOrElse("chkptDir","");
    var chkptIter = options.getOrElse("chkptIter","100").toInt

    // algorithm-specific inputs
    val lambda = options.getOrElse("lambda", "0.01").toDouble // regularization parameter
    val numRounds = options.getOrElse("numRounds", "20").toInt // number of outer iterations, called T in the paper
    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble; // fraction of local points to be processed per round, H = localIterFrac * n
    val beta = options.getOrElse("beta","1.0").toDouble;  // scaling parameter when combining the updates of the workers (1=averaging for CoCoA)
    val gamma = options.getOrElse("gamma","1.0").toDouble;  // aggregation parameter for CoCoA+ (1=adding, 1/K=averaging)
    val debugIter = options.getOrElse("debugIter","1").toInt // set to -1 to turn off debugging output
    val seed = options.getOrElse("seed","0").toInt // set seed for debug purposes

    // start spark context
    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)
    if (chkptDir != "") {
      sc.setCheckpointDir(chkptDir)
    } else {
      chkptIter = numRounds + 1
    }
    sc.setLogLevel("ERROR")

    // read in data
    val data = OptUtils.loadLIBSVMData(sc, dataFile, numSplits, numFeatures).cache()
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainData = splits(0).cache()
    val testData = splits(1)

    val n = data.count().toInt // number of data examples

    // compute H, # of local iterations
    var localIters = (localIterFrac * n / trainData.partitions.size).toInt
    localIters = Math.max(localIters,1)

    // for the primal-dual algorithms to run correctly, the initial primal vector has to be zero
    // (corresponding to dual alphas being zero)
    val wInit = DenseVector.zeros[Double](numFeatures)

    // set to solve hingeloss SVM
    val loss = OptUtils.hingeLoss _
    val params = Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
    val debug = DebugParams(testData, debugIter, seed, chkptIter)


//     run CoCoA+
    val (finalwCoCoAPlus, finalalphaCoCoAPlus) = CoCoA.runCoCoA(trainData, params, debug, plus=true)
    OptUtils.printSummaryStatsPrimalDual("CoCoA+", trainData, finalwCoCoAPlus, finalalphaCoCoAPlus, lambda, testData)

    // run CoCoA
    val (finalwCoCoA, finalalphaCoCoA) = CoCoA.runCoCoA(trainData, params, debug, plus=false)
    OptUtils.printSummaryStatsPrimalDual("CoCoA", trainData, finalwCoCoA, finalalphaCoCoA, lambda, testData)

    sc.stop()
  }
}
