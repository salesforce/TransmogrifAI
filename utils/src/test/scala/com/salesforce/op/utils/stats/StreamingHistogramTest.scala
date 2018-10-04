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

  it should "approximate Gaussian distribution CDF" in {
    val grid1 = linspace(-3, 3, 25)
    val grid2 = linspace(-1, 11, 25)
    val grid3 = grid1 ++ grid2
    val gaussian1 = Gaussian(0, 1)
    val gaussian2 = Gaussian(5, 5)
    val mixture = MixtureDistribution(gaussian1, gaussian2, 0.8)
    val sampleSize = 500

    cdfTest("Gaussian(0, 1)", gaussian1, grid1, sampleSize, 100, 50)
    cdfTest("Gaussian(0, 1)", gaussian1, grid1, sampleSize, 200, 50)
    cdfTest("Gaussian(0, 1)", gaussian1, grid1, sampleSize, 400, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, grid2, sampleSize, 100, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, grid2, sampleSize, 200, 50)
    cdfTest("Gaussian(5, 5)", gaussian2, grid2, sampleSize, 400, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, grid3, sampleSize, 100, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, grid3, sampleSize, 200, 50)
    cdfTest("0.8 * Gaussian(0, 1) + 0.2 * Gaussian(5, 5)", mixture, grid3, sampleSize, 400, 50)
  }

  it should "approximate Gamma distribution CDF" in {
    val grid1 = linspace(0.0001, 4, 25)
    val grid2 = linspace(950, 1050, 25)
    val grid3 = grid1 ++ linspace(4, 950, 25) ++ grid2
    val gamma1 = new Gamma(20, 1.0 / 2)
    val gamma2 = new Gamma(1000000, 1.0 / 1000)
    val mixture = MixtureDistribution(gamma1, gamma2, 0.95)
    val sampleSize = 500

    cdfTest("Gamma(20, 0.5)", gamma1, grid1, sampleSize, 100, 50)
    cdfTest("Gamma(20, 0.5)", gamma1, grid1, sampleSize, 200, 50)
    cdfTest("Gamma(20, 0.5)", gamma1, grid1, sampleSize, 400, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, grid2, sampleSize, 100, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, grid2, sampleSize, 200, 50)
    cdfTest("Gamma(1000000, 0.001)", gamma2, grid2, sampleSize, 400, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, grid3, sampleSize, 100, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, grid3, sampleSize, 200, 50)
    cdfTest("0.95 * Gamma(20, 0.5) + 0.05 * Gamma(1000000, 0.001)", mixture, grid3, sampleSize, 400, 50)
  }

  it should "approximate Beta distribution CDF" in {
    val grid = linspace(0.0001, 0.999, 25)
    val beta1 = new Beta(5, 1)
    val beta2 = new Beta(1, 5)
    val mixture = MixtureDistribution(beta1, beta2, 0.5)
    val sampleSize = 500

    cdfTest("Beta(5, 1)", beta1, grid, sampleSize, 100, 50)
    cdfTest("Beta(5, 1)", beta1, grid, sampleSize, 200, 50)
    cdfTest("Beta(5, 1)", beta1, grid, sampleSize, 400, 50)
    cdfTest("Beta(1, 5)", beta2, grid, sampleSize, 100, 50)
    cdfTest("Beta(1, 5)", beta2, grid, sampleSize, 200, 50)
    cdfTest("Beta(1, 5)", beta2, grid, sampleSize, 400, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, grid, sampleSize, 100, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, grid, sampleSize, 200, 50)
    cdfTest("0.5 * Beta(5, 1) + 0.5 * Beta(1, 5)", mixture, grid, sampleSize, 400, 50)
  }

  private def cdfTest(
    distributionName: String,
    dist: Distribution[Double] with HasCdf,
    grid: Array[Double],
    sampleSize: Int,
    maxBins: Int,
    numResults: Int): Unit = {

    def getResults(sampleSize: Int, maxBins: Int): Array[DistributionTestResult] = {
      val sample = (0 until sampleSize).map(_ => dist.draw()).toArray
      val hist = new StreamingHistogram(maxBins)
      val equiDist = equiDistBins(sample, maxBins)

      hist.update(sample: _*)

      grid.map { gridPoint =>
        DistributionTestResult(
          point = gridPoint,
          trueCDF = dist.cdf(gridPoint),
          streamingSumCDF = hist.sumCDF(gridPoint),
          streamingEmpiricalCDF = hist.empiricalCDF(gridPoint),
          equiDistSumCDF = StreamingHistogram.sumCDF(equiDist, gridPoint),
          equiDistEmpiricalCDF = StreamingHistogram.empiricalCDF(equiDist, gridPoint))
      }
    }

    def cdfMSE(allResults: Array[Array[DistributionTestResult]]): Array[MSEResult] = {
      val points = allResults.flatMap(_.map(_.point)).distinct.sorted

      def computeMSE(results: Array[DistributionTestResult], f: DistributionTestResult => Double): Double = {
        val bias = breeze.stats.mean(results.map(r => r.trueCDF - f(r)))
        val variance = breeze.stats.variance(results.map(f))

        math.pow(bias, 2) + variance
      }

      points.map { pt =>
        val results = allResults.map(_.filter(_.point == pt)(0))

        MSEResult(
          streamingSumCDFMSE = computeMSE(results, _.streamingSumCDF),
          streamingEmpiricalCDFMSE = computeMSE(results, _.streamingEmpiricalCDF),
          equiDistSumCDFMSE = computeMSE(results, _.equiDistSumCDF),
          equiDistEmpiricalCDFMSE = computeMSE(results, _.equiDistEmpiricalCDF))
      }
    }

    val mses = cdfMSE((0 until numResults).map(_ => getResults(sampleSize, maxBins)).toArray)

    val MeanAndVariance(streamingSumCDFMSEMean, streamingSumCDFMSEVar, _) =
      meanAndVariance(mses.map(_.streamingSumCDFMSE))
    val MeanAndVariance(streamingEmpiricalCDFMSEMean, streamingEmpiricalCDFMSEVar, _) =
      meanAndVariance(mses.map(_.streamingEmpiricalCDFMSE))
    val MeanAndVariance(equiDistSumCDFMSEMean, equiDistSumCDFMSEVar, _) =
      meanAndVariance(mses.map(_.equiDistSumCDFMSE))
    val MeanAndVariance(equiDistEmpiricalCDFMSEMean, equiDistEmpiricalCDFMSEVar, _) =
      meanAndVariance(mses.map(_.equiDistEmpiricalCDFMSE))

    println("-" * 50)
    println(s"Checking distribution $distributionName " +
      s"[bins = $maxBins, sample size = $sampleSize, iterations = $numResults]")
    println("-" * 50)
    println(s"Streaming histogram sum CDF MSE mean and variance: $streamingSumCDFMSEMean, $streamingSumCDFMSEVar")
    println("Streaming histogram empirical CDF MSE mean and variance: " +
      s"$streamingEmpiricalCDFMSEMean, $streamingEmpiricalCDFMSEVar")
    println(s"Equidistant histogram sum CDF MSE mean and variance: $equiDistSumCDFMSEMean, $equiDistSumCDFMSEVar")
    println("Equidistant histogram empirical CDF MSE mean and variance: " +
      s"$equiDistEmpiricalCDFMSEMean, $equiDistEmpiricalCDFMSEVar")

    val mseMeans = Array(
      streamingSumCDFMSEMean, streamingEmpiricalCDFMSEMean, equiDistSumCDFMSEMean, equiDistEmpiricalCDFMSEMean)

    for {
      x1 <- mseMeans
      x2 <- mseMeans
    } yield {
      math.abs(x1 - x2) <= 0.001 shouldBe true
    }
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

  type Distribution[T] = Density[T] with Rand[T]

  case class MixtureDistribution(
      d1: Distribution[Double] with HasCdf,
      d2: Distribution[Double]with HasCdf,
      p: Double) extends Density[Double] with Rand[Double] with HasCdf {
      val bernoulli = new Bernoulli(p)
      def apply(x: Double): Double = mixture(d1(x), d2(x))
      def cdf(x: Double): Double = mixture(d1.cdf(x), d2.cdf(x))
      def draw(): Double = if (bernoulli.draw) d1.draw else d2.draw
      def probability(x: Double, y: Double): Double = mixture(d1.probability(x, y), d2.probability(x, y))

      private def mixture(x: Double, y: Double): Double = p * x + (1 - p) * x
  }

  case class DistributionTestResult(
      point: Double,
      trueCDF: Double,
      streamingSumCDF: Double,
      streamingEmpiricalCDF: Double,
      equiDistSumCDF: Double,
      equiDistEmpiricalCDF: Double)

  case class MSEResult(
      streamingSumCDFMSE: Double,
      streamingEmpiricalCDFMSE: Double,
      equiDistSumCDFMSE: Double,
      equiDistEmpiricalCDFMSE: Double)
}
