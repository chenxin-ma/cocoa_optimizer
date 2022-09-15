package org.example

import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by hubo on 2018/1/13
 */
object WordCount {
  def main(args: Array[String]) {
    var masterUrl = "local"
    var inputPath = "/Users/chenxin.ma/Documents/projects/hello_cocoa_mvn2/pom.xml"
    var outputPath = "/Users/chenxin.ma/Documents/projects/hello_cocoa_mvn2/1.txt"

    if (args.length == 1) {
      masterUrl = args(0)
    } else if (args.length == 3) {
      masterUrl = args(0)
      inputPath = args(1)
      outputPath = args(2)
    }

    println(s"masterUrl:$masterUrl, inputPath: $inputPath, outputPath: $outputPath")
    val sparkConf = new SparkConf().setMaster(masterUrl).setAppName("WordCount")
    val sc = new SparkContext(sparkConf)

    val rowRdd = sc.textFile(inputPath)
    val resultRdd = rowRdd.flatMap(line => line.split("\\s+"))
      .map(word => (word, 1)).reduceByKey(_ + _)

    resultRdd.saveAsTextFile(outputPath)
  }
}
