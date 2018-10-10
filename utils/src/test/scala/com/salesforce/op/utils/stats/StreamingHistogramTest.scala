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

import breeze.integrate
import breeze.stats.{meanAndVariance, MeanAndVariance}
import breeze.stats.distributions._
import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.stats.StreamingHistogramTest._
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class StreamingHistogramTest extends FlatSpec with TestCommon {

  val histogramSampleSize = 500
  val mcSampleSize = 1000

  Spec(classOf[StreamingHistogram]) should "produce correct histogram distribution" in {
    referenceHistogram.getBins.map {
      case (point, count) => round(point) -> count
    } should contain theSameElementsAs Seq(2.0 -> 1L, 9.5 -> 2L, 19.33 -> 3L, 32.67 -> 3L, 45 -> 1L)
  }

  it should "compute sum algorithm correctly" in {
    val hist = referenceHistogram

    round(hist.sum(0)) shouldEqual 0.0
    round(hist.sum(2)) shouldEqual 0.5
    round(hist.sum(9.5)) shouldEqual 2.0
    round(hist.sum(15)) shouldEqual 3.28
    round(hist.sum(20)) shouldEqual 4.65
    round(hist.sum(35)) shouldEqual 8.03
    round(hist.sum(45)) shouldEqual 9.5
    round(hist.sum(46)) shouldEqual 10.0
  }

  it should "compute the correct density" in {
    val hist = referenceHistogram

    round(hist.density(1)) shouldEqual 0.0
    round(hist.density(1.999)) shouldEqual 0.05
    round(hist.density(5)) shouldEqual 0.15
    round(hist.density(12)) shouldEqual 0.25
    round(hist.density(25)) shouldEqual 0.3
    round(hist.density(35)) shouldEqual 0.2
    round(hist.density(45.001)) shouldEqual 0.05
    round(hist.density(46)) shouldEqual 0.0
  }

  it should "correctly approximate standard normal distribution" in {
    val sampleSize = 500
    val gaussian1 = Gaussian(0, 1)
    val gaussian2 = Gaussian(5, 5)
    val mixture = MixtureDistribution(gaussian1, gaussian2, 0.8, sampleSize)

    cdfTest("Gaussian(0, 1)", gaussian1, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Gaussian(0, 1)", gaussian1, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Gaussian(0, 1)", gaussian1, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, histogramSampleSize, 400, mcSampleSize, 50)
  }

  it should "approximate Gamma distribution CDF" in {
    val sampleSize = 500
    val gamma1 = new Gamma(20, 1.0 / 2)
    val gamma2 = new Gamma(1000000, 1.0 / 1000)
    val mixture = MixtureDistribution(gamma1, gamma2, 0.95, sampleSize)

    cdfTest("Gamma(20, 0.5)", gamma1, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Gamma(20, 0.5)", gamma1, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Gamma(20, 0.5)", gamma1, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, histogramSampleSize, 400, mcSampleSize, 50)
  }

  it should "approximate Beta distribution CDF" in {
    val sampleSize = 500
    val beta1 = new Beta(5, 1)
    val beta2 = new Beta(1, 5)
    val mixture = MixtureDistribution(beta1, beta2, 0.5, sampleSize)

    cdfTest("Beta(5, 1)", beta1, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Beta(5, 1)", beta1, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Beta(5, 1)", beta1, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("Beta(1, 5)", beta2, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("Beta(1, 5)", beta2, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("Beta(1, 5)", beta2, histogramSampleSize, 400, mcSampleSize, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, histogramSampleSize, 100, mcSampleSize, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, histogramSampleSize, 200, mcSampleSize, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, histogramSampleSize, 400, mcSampleSize, 50)
  }


  private def cdfTest(
    distributionName: String,
    dist: ContinuousDistr[Double] with HasCdf,
    histogramSampleSize: Int,
    maxBins: Int,
    mcSampleSize: Int,
    numResults: Int): MSEDistributionResult = {
    val sample = dist.sample(histogramSampleSize).toArray
    val hist = new StreamingHistogram(maxBins)
    val equiDist = equiDistBins(sample, maxBins)

    hist.update(sample: _*)

    val mses: Array[MSEResult] = (0 until numResults).map { _ =>
      val mcSample = dist.sample(mcSampleSize)
      val trueDensities = mcSample.map(dist.pdf).toArray
      val trueCDFs = mcSample.map(dist.cdf).toArray

      def computeMSE(f: Double => Double, trueValues: Array[Double]): Double =
        mcSample.map(f).zip(trueValues).map { case (a, b) => math.pow(a - b, 2) }.sum / mcSampleSize

      MSEResult(
        streamingDensityMSE = computeMSE(hist.density(_), trueDensities),
        streamingSumCDFMSE = computeMSE(hist.sumCDF(_), trueCDFs),
        streamingEmpiricalCDFMSE = computeMSE(hist.empiricalCDF(_), trueCDFs),
        equiDistDensityMSE = computeMSE(StreamingHistogram.density(equiDist, _), trueDensities),
        equiDistSumCDFMSE = computeMSE(StreamingHistogram.sumCDF(equiDist, _), trueCDFs),
        equiDistEmpiricalCDFMSE = computeMSE(StreamingHistogram.empiricalCDF(equiDist, _), trueCDFs))
    }.toArray

    val result = MSEDistributionResult(
      streamingDensityMSE = meanAndVariance(mses.map(_.streamingDensityMSE)),
      streamingSumCDFMSE = meanAndVariance(mses.map(_.streamingSumCDFMSE)),
      streamingEmpiricalCDFMSE = meanAndVariance(mses.map(_.streamingEmpiricalCDFMSE)),
      equiDistDensityMSE = meanAndVariance(mses.map(_.equiDistDensityMSE)),
      equiDistSumCDFMSE = meanAndVariance(mses.map(_.equiDistSumCDFMSE)),
      equiDistEmpiricalCDFMSE = meanAndVariance(mses.map(_.equiDistEmpiricalCDFMSE)))

    println("-" * 50)
    println(s"Checking distribution $distributionName " +
      s"[bins = $maxBins, sample size = $histogramSampleSize, iterations = $numResults]")
    println("-" * 50)
    println("Streaming histogram density MSE mean and variance: " +
      s"${result.streamingDensityMSE.mean}, ${result.streamingDensityMSE.variance}")
    println("Streaming histogram sum CDF MSE mean and variance: " +
      s"${result.streamingSumCDFMSE.mean}, ${result.streamingSumCDFMSE.variance}")
    println("Streaming histogram empirical CDF MSE mean and variance: " +
      s"${result.streamingEmpiricalCDFMSE.mean}, ${result.streamingEmpiricalCDFMSE.variance}")
    println(s"Equidistant histogram density MSE mean and variance: " +
      s"${result.equiDistDensityMSE.mean}, ${result.equiDistDensityMSE.variance}")
    println(s"Equidistant histogram sum CDF MSE mean and variance: " +
      s"${result.equiDistSumCDFMSE.mean}, ${result.equiDistSumCDFMSE.variance}")
    println("Equidistant histogram empirical CDF MSE mean and variance: " +
      s"${result.equiDistEmpiricalCDFMSE.mean}, ${result.equiDistEmpiricalCDFMSE.variance}")

    result
  }

  private def equiDistBins(points: Array[Double], numBins: Int): Array[(Double, Double)] = {
    val a = points.min - 0.001
    val b = points.max + 0.001

    linspace(a, b, numBins).sliding(2).map(_ match {
      case Array(p, q) => ((p + q) / 2, points.filter(d => d >= p && d < q).length.toDouble)
    }).toArray
  }

  private def linspace(a: Double, b: Double, n: Int): Array[Double] =
    (0 to n).map(k => a + ((b - a) * k) / n).toArray

  private def referenceHistogram: StreamingHistogram = {
    val hist = new StreamingHistogram(5)
    hist.update(23, 19, 10, 16, 36)
    hist.getBins should contain theSameElementsAs Seq(23.0, 19.0, 10.0, 16.0, 36.0).map(_ -> 1L)
    hist.update(2)
    hist.getBins should contain theSameElementsAs Seq(2.0, 10.0, 23.0, 36.0).map(_ -> 1L) ++ Seq(17.5 -> 2L)
    hist.update(9)
    hist.getBins should contain theSameElementsAs Seq(2.0, 23.0, 36.0).map(_ -> 1L) ++ Seq(9.5, 17.5).map(_ -> 2L)

    val hist2 = new StreamingHistogram(5)
    hist.update(32, 30, 45)

    hist.merge(hist2)
  }

  private def round(x: Double): Double = math.round(x * 100).toDouble / 100
}

object StreamingHistogramTest {

  case class MixtureDistribution(
      d1: ContinuousDistr[Double] with HasCdf,
      d2: ContinuousDistr[Double]with HasCdf,
      p: Double,
      mcSampleSize: Int) extends ContinuousDistr[Double] with HasCdf {
      val bernoulli = new Bernoulli(p)

      // Required for defining class, but don't use for test purposes
      def logNormalizer: Double = 0.0
      def unnormalizedLogPdf(x: Double): Double = 0.0
      def probability(x: Double,y: Double): Double = 0.0

      override def cdf(x: Double): Double = mixture(d1.cdf(x), d2.cdf(x))
      override def pdf(x: Double): Double = mixture(d1.pdf(x), d2.pdf(x))
      override def draw(): Double = if (bernoulli.draw) d1.draw else d2.draw

      def entropy: Double = (0 until mcSampleSize).map { _ =>
        val samp = draw()
        apply(samp) * math.log(samp)
      }.sum / mcSampleSize

      private def mixture(x: Double, y: Double): Double = p * x + (1 - p) * x
  }

  case class MSEResult(
      streamingDensityMSE: Double,
      streamingSumCDFMSE: Double,
      streamingEmpiricalCDFMSE: Double,
      equiDistDensityMSE: Double,
      equiDistSumCDFMSE: Double,
      equiDistEmpiricalCDFMSE: Double)

  case class MSEDistributionResult(
      streamingDensityMSE: MeanAndVariance,
      streamingSumCDFMSE: MeanAndVariance,
      streamingEmpiricalCDFMSE: MeanAndVariance,
      equiDistDensityMSE: MeanAndVariance,
      equiDistSumCDFMSE: MeanAndVariance,
      equiDistEmpiricalCDFMSE: MeanAndVariance)
}
