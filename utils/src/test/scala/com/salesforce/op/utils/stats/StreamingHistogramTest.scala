/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.stats

import breeze.stats.{meanAndVariance, MeanAndVariance}
import breeze.stats.distributions._
import com.salesforce.op.test.TestSparkContext
import com.salesforce.op.utils.stats.RichStreamingHistogram._
import com.salesforce.op.utils.stats.StreamingHistogram.StreamingHistogramBuilder
import com.salesforce.op.utils.stats.StreamingHistogramTest._
import org.apache.log4j.Logger
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class StreamingHistogramTest extends FlatSpec with TestSparkContext {

  val testPadding = 0.5
  val logger = Logger.getLogger(getClass)
  val histogramSampleSize = 1000
  val mcSampleSize = 1000
  val numResults = 5

  // Enforce Kryo serialization check
  conf.set("spark.kryo.registrationRequired", "true")

  Spec(classOf[StreamingHistogram]) should "produce correct histogram distribution" in {
    referenceHistogram.getPaddedBins(testPadding).map {
      case (point, count) => round(point) -> count
    } should contain theSameElementsAs
      Seq(1.5 -> 0.0, 2.0 -> 1L, 9.5 -> 2L, 19.33 -> 3L, 32.67 -> 3L, 45 -> 1L, 45.5 -> 0.0)
  }

  it should "compute sum algorithm correctly" in {
    val hist = referenceHistogram

    round(hist.sum(0)) shouldEqual 0.0
    round(hist.sum(2)) shouldEqual 0.5
    round(hist.sum(9.5)) shouldEqual 2.0
    round(hist.sum(15)) shouldEqual 3.28
    round(hist.sum(20)) shouldEqual 4.65
    round(hist.sum(35)) shouldEqual 8.03
    round(hist.sum(45)) shouldEqual 10.0
    round(hist.sum(46)) shouldEqual 10.0
  }

  it should "work with spark" in {
    val data = sc.parallelize((0 to 10).map(_.toDouble), 2)
    val histogram = {
      val seqOp = (bldr: StreamingHistogramBuilder, point: Double) => {
        bldr.update(point)
        bldr
      }
      val combOp = (bldr1: StreamingHistogramBuilder, bldr2: StreamingHistogramBuilder) => {
        bldr1.merge(bldr2.build)
        bldr1
      }

      data.aggregate(new StreamingHistogramBuilder(15, 500, 1))(seqOp, combOp).build
    }

    histogram.getPaddedBins(testPadding) should contain theSameElementsAs
      (0 to 10).map(k => (k.toDouble, 1.0)) ++ Array(-0.5 -> 0.0, 10.5 -> 0.0)
  }

  it should "yield correct histogram density estimator" in {
    val builder = new StreamingHistogramBuilder(10, 500, 1)

    Array(0.0 -> 1L, 2.0 -> 3L, 3.0 -> 3L, 4.0 -> 1L).foreach { case (pt, ct) => builder.update(pt, ct) }

    val pdf = builder.build.density(testPadding)

    pdf(-1.0) shouldEqual 0.0
    pdf(-0.5) shouldEqual 0.0625
    pdf(0.0) shouldEqual 0.25
    pdf(1.0) shouldEqual 0.25
    pdf(2.0) shouldEqual 0.375
    pdf(2.5) shouldEqual 0.375
    pdf(3.0) shouldEqual 0.25
    pdf(3.5) shouldEqual 0.25
    pdf(4.0) shouldEqual 0.0625
    pdf(4.5) shouldEqual 0.0
    pdf(5.0) shouldEqual 0.0
  }

  // Checks that it does well for well-behaved distributions
  it should "correctly approximate standard normal distribution" in {
    val sampleSize = 500
    val gaussian = Gaussian(0, 1)(RandBasis.mt0)
    val distributionName = "Gaussian(0, 1)"

    val result75 =
      distributionTestResult(distributionName, gaussian, histogramSampleSize, 75, mcSampleSize, numResults)
    val result125 =
      distributionTestResult(distributionName, gaussian, histogramSampleSize, 125, mcSampleSize, numResults)
    val result250 =
      distributionTestResult(distributionName, gaussian, histogramSampleSize, 250, mcSampleSize, numResults)

    result75.streamingDensityMSE.mean should be >= result75.equiDistDensityMSE.mean
    result75.absoluteMeanDiff should be < 0.01
    result125.streamingDensityMSE.mean should be >= result125.equiDistDensityMSE.mean
    result125.absoluteMeanDiff should be < 0.01
    result250.streamingDensityMSE.mean should be >= result250.equiDistDensityMSE.mean
    result250.absoluteMeanDiff should be < 0.01
  }

  // Check that it does better for distributions with outliers
  it should "better approximate distribution with large outliers" in {
    val sampleSize = 500
    val gamma1 = new Gamma(20, 1.0 / 2)(RandBasis.mt0)
    val gamma2 = new Gamma(1000000, 1.0 / 1000)(RandBasis.mt0)
    val mixture = MixtureDistribution(gamma1, gamma2, 0.95, sampleSize)
    val distributionName = "0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)"

    val result75 =
      distributionTestResult(distributionName, mixture, histogramSampleSize, 75, mcSampleSize, numResults)
    val result125 =
      distributionTestResult(distributionName, mixture, histogramSampleSize, 125, mcSampleSize, numResults)
    val result250 =
      distributionTestResult(distributionName, mixture, histogramSampleSize, 250, mcSampleSize, numResults)

    result75.streamingDensityMSE.mean should be <= result75.equiDistDensityMSE.mean
    result75.absoluteMeanDiff should be > 0.05
    result75.absoluteMeanDiff should be < 0.06
    result125.streamingDensityMSE.mean should be <= result125.equiDistDensityMSE.mean
    result125.absoluteMeanDiff should be > 0.06
    result125.absoluteMeanDiff should be < 0.07
    result250.streamingDensityMSE.mean should be <= result250.equiDistDensityMSE.mean
    result250.absoluteMeanDiff should be > 0.04
    result250.absoluteMeanDiff should be < 0.05
  }

  private def distributionTestResult(
    distributionName: String,
    dist: ContinuousDistr[Double],
    histogramSampleSize: Int,
    maxBins: Int,
    mcSampleSize: Int,
    numResults: Int): MSEDistributionResult = {

    val sample = dist.sample(histogramSampleSize).toArray
    val hist = new StreamingHistogramBuilder(maxBins, 100, 1)

    sample.foreach(hist.update(_))

    val equiDistBins = equiDistantBins(sample, maxBins)

    val mses: Array[MSEResult] = (0 until numResults).map { _ =>
      val mcSample = dist.sample(mcSampleSize)
      val trueDensities = mcSample.map(dist.pdf).toArray

      def computeMSE(f: Double => Double, trueValues: Array[Double]): Double =
        mcSample.map(f).zip(trueValues).map { case (a, b) => math.pow(a - b, 2) }.sum / mcSampleSize

      MSEResult(
        streamingDensityMSE = computeMSE(hist.build.density(testPadding), trueDensities),
        equiDistDensityMSE = computeMSE(RichStreamingHistogram.density(equiDistBins), trueDensities))
    }.toArray

    val result = MSEDistributionResult(
      streamingDensityMSE = meanAndVariance(mses.map(_.streamingDensityMSE)),
      equiDistDensityMSE = meanAndVariance(mses.map(_.equiDistDensityMSE)))

    logger.info(s"\n${
      ("-" * 50)  + "\n" +
        s"Checking distribution $distributionName " +
        s"[bins = $maxBins, sample size = $histogramSampleSize, iterations = $numResults]\n" +
        ("-" * 50) + s"\n$result"
    }")

    result
  }

  private def referenceHistogram: StreamingHistogram = {
    val hist = new StreamingHistogramBuilder(5, 0, 1)

    Seq(23, 19, 10, 16, 36, 2, 9).foreach(hist.update(_))

    val hist2 = new StreamingHistogramBuilder(5, 0, 1)
    Seq(32, 30, 45).foreach(hist2.update(_))

    hist.merge(hist2.build)

    hist.build
  }

  private def equiDistantBins(points: Array[Double], numBins: Int): Array[(Double, Double)] =
    if (points.isEmpty) Array()
    else {
      val mainBins = points match {
        case Array(p) => Array((p, 1.0))
        case arr =>
          val (min, max) = (arr.min, arr.max)

          linspace(min, max, numBins).sliding(2).map {
            case Array(p, q) =>
              (p + q) / 2 -> points.filter(x => x >= p && x < q).length.toDouble
          }.toArray
        }

      RichStreamingHistogram.paddedBins(mainBins, 0.1)
    }

  private def linspace(a: Double, b: Double, n: Int): Array[Double] =
    (0 to n).map(k => a + (b - a) * k / n).toArray

  private def round(x: Double): Double = math.round(x * 100).toDouble / 100
}

object StreamingHistogramTest {

  case class MixtureDistribution(
      d1: ContinuousDistr[Double],
      d2: ContinuousDistr[Double],
      p: Double,
      mcSampleSize: Int) extends ContinuousDistr[Double] {
      val bernoulli = new Bernoulli(p)(RandBasis.mt0)

      // Required for defining class, but don't use for test purposes
      def logNormalizer: Double = 0.0
      def unnormalizedLogPdf(x: Double): Double = 0.0
      def probability(x: Double, y: Double): Double = 0.0

      override def pdf(x: Double): Double = mixture(d1.pdf(x), d2.pdf(x))
      override def draw(): Double = if (bernoulli.draw) d1.draw else d2.draw

      private def mixture(x: Double, y: Double): Double = p * x + (1 - p) * x
  }

  case class MSEResult(
      streamingDensityMSE: Double,
      equiDistDensityMSE: Double)

  case class MSEDistributionResult(
      streamingDensityMSE: MeanAndVariance,
      equiDistDensityMSE: MeanAndVariance) {
    override def toString(): String =
      "Streaming histogram density MSE mean and variance: " +
        s"${streamingDensityMSE.mean}, ${streamingDensityMSE.variance}\n" +
        "Equidistant histogram density MSE mean and variance: " +
        s"${equiDistDensityMSE.mean}, ${equiDistDensityMSE.variance}"

    def absoluteMeanDiff: Double = math.abs(streamingDensityMSE.mean - equiDistDensityMSE.mean)
  }
}
