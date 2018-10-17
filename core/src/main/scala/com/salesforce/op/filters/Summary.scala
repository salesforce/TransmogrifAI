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

package com.salesforce.op.filters

import java.util.concurrent.atomic.AtomicReference
import java.util.function.BinaryOperator

import com.salesforce.op.stages.impl.feature.HashAlgorithm
import com.salesforce.op.utils.stats.RichStreamingHistogram._
import com.salesforce.op.utils.stats.StreamingHistogram.StreamingHistogramBuilder
import com.twitter.algebird.{Monoid, Semigroup}
import org.apache.spark.mllib.feature.HashingTF

import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap

/**
 * Class used to get summaries of prepared features to determine distribution binning strategy
 *
 * @param min   minimum value seen for double, minimum number of tokens in one text for text
 * @param max   maximum value seen for double, maximum number of tokens in one text for text
 * @param sum   sum of values for double, total number of tokens for text
 * @param count number of doubles for double, number of texts for text
 */
case class Summary(min: Double, max: Double, sum: Double, count: Double)

case object Summary {

  val empty: Summary = Summary(Double.PositiveInfinity, Double.NegativeInfinity, 0.0, 0.0)

  implicit val monoid: Monoid[Summary] = new Monoid[Summary] {
    override def zero = Summary.empty
    override def plus(l: Summary, r: Summary) = Summary(
      math.min(l.min, r.min), math.max(l.max, r.max), l.sum + r.sum, l.count + r.count
    )
  }

  /**
   * @param preppedFeature processed feature
   * @return feature summary derived from processed feature
   */
  def apply(preppedFeature: ProcessedSeq): Summary = {
    preppedFeature match {
      case Left(v) => Summary(v.size, v.size, v.size, 1.0)
      case Right(v) => monoid.sum(v.map(d => Summary(d, d, d, 1.0)))
    }
  }
}

class TextSummary(textFormula: TextSummary => Int) {

  private[this] val count: AtomicReference[Double] = new AtomicReference(0)
  private[this] val distribution: HashMap[Double, Double] = HashMap()
  private[this] var hashingTFOpt: Option[HashingTF] = None
  private[this] val maxTokens: AtomicReference[Double] = new AtomicReference(Double.NegativeInfinity)
  private[this] val minTokens: AtomicReference[Double] = new AtomicReference(Double.PositiveInfinity)
  private[this] val numTokens: AtomicReference[Double] = new AtomicReference(0)
  private[this] val maxOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = math.max(s, t)
  }
  private[this] val minOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = math.min(s, t)
  }
  private[this] val sumOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = s + t
  }

  final def getCount(): Double = count.get

  final def getDistribution(): Array[(Double, Double)] = distribution.toArray.sortBy(_._1)

  final def getFeatureDistribution(featureKey: FeatureKey, totalCount: Double): FeatureDistribution = {
    val thisCount = getCount
    val nullCount = totalCount - thisCount
    val dist = getDistribution

    FeatureDistribution(
      name = featureKey._1,
      key = featureKey._2,
      count = thisCount.toLong,
      nulls = nullCount.toLong,
      distribution = dist.map(_._2),
      summaryInfo = dist.map(_._1))
  }

  final def getNumTokens(): Double = numTokens.get

  final def getMinTokens(): Double = minTokens.get

  final def getMaxTokens(): Double = maxTokens.get

  final def merge(other: TextSummary): this.type = synchronized {
    count.accumulateAndGet(other.getCount, sumOp)
    maxTokens.accumulateAndGet(other.getMaxTokens, maxOp)
    minTokens.accumulateAndGet(other.getMinTokens, minOp)
    numTokens.accumulateAndGet(other.getNumTokens, sumOp)

    this
  }

  final def mergeDistribution(other: TextSummary): this.type = {
    other.getDistribution.foreach { case (point, count) =>
      val newCount = distribution.get(point).getOrElse(0.0) + count
      distribution += (point -> newCount)
    }

    this
  }

  final def setHashingTF(): this.type = synchronized {
    hashingTFOpt = Option {
      new HashingTF(numFeatures = textFormula(this)).setBinary(false)
        .setHashAlgorithm(HashAlgorithm.MurMur3.entryName.toLowerCase)
    }

    this
  }

  final def toSummary: Summary = Summary(
    min = getMinTokens,
    max = getMaxTokens,
    sum = getNumTokens,
    count = getCount)

  final def update(text: Seq[String]): this.type = synchronized {
    val size: Double = text.length
    count.accumulateAndGet(1.0, sumOp)
    maxTokens.accumulateAndGet(size, maxOp)
    minTokens.accumulateAndGet(size, minOp)
    numTokens.accumulateAndGet(size, sumOp)

    this
  }

  final def updateDistribution(text: Seq[String]): this.type = synchronized {
    hashingTFOpt match {
      case Some(hashingTF) =>
        val points: Array[Double] = hashingTF.transform(text).toArray
        val currentCounts: Seq[Double] = points.map(distribution.get(_).getOrElse(0.0))

        points.zip(currentCounts).foreach { case (point, count) =>
          distribution += point -> (count + 1.0)
        }
      case None =>
        throw new RuntimeException("HashingTF must be set in order to update text summary distribution")

    }

    this
  }
}

object TextSummary {
  implicit val semigroup = new Semigroup[TextSummary] {
    def plus(l: TextSummary, r: TextSummary): TextSummary = l.merge(r)
  }
}

class HistogramSummary(maxBins: Int, maxSpoolSize: Int) {

  private[this] val builder: StreamingHistogramBuilder = new StreamingHistogramBuilder(maxBins, maxSpoolSize, 1)
  private[this] val count: AtomicReference[Double] = new AtomicReference(0)
  private[this] val maximum: AtomicReference[Double] = new AtomicReference(Double.NegativeInfinity)
  private[this] val minimum: AtomicReference[Double] = new AtomicReference(Double.PositiveInfinity)
  private[this] val valueSum: AtomicReference[Double] = new AtomicReference(0)
  private[this] val maxOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = math.max(s, t)
  }
  private[this] val minOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = math.min(s, t)
  }
  private[this] val sumOp: BinaryOperator[Double] = new BinaryOperator[Double] {
    def apply(s: Double, t: Double): Double = s + t
  }

  final def getCount(): Double = count.get

  final def getDistribution(): Array[(Double, Double)] = builder.build.getBins

  final def getFeatureDistribution(featureKey: FeatureKey, totalCount: Double): FeatureDistribution = {
    val thisCount = getCount
    val nullCount = totalCount - thisCount
    val dist = getDistribution

    FeatureDistribution(
      name = featureKey._1,
      key = featureKey._2,
      count = thisCount.toLong,
      nulls = nullCount.toLong,
      distribution = dist.map(_._2),
      summaryInfo = dist.map(_._1))
  }

  final def getMaximum(): Double = maximum.get

  final def getMinimum(): Double = minimum.get

  final def getValueSum(): Double = valueSum.get

  final def merge(other: HistogramSummary): this.type = synchronized {
    maximum.accumulateAndGet(other.getMaximum, maxOp)
    minimum.accumulateAndGet(other.getMinimum, minOp)
    count.accumulateAndGet(other.getCount, sumOp)
    other.getDistribution.foreach { case (pt, count) =>
      valueSum.accumulateAndGet(pt, sumOp)
      builder.update(pt, count.toLong)
    }

    this
  }

  final def toSummary: Summary = Summary(
    min = getMinimum,
    max = getMaximum,
    sum = getValueSum,
    count = getCount)

  final def update(points: Seq[Double]): this.type = synchronized {
    points.foreach { pt =>
      maximum.accumulateAndGet(pt, maxOp)
      minimum.accumulateAndGet(pt, minOp)
      valueSum.accumulateAndGet(pt, sumOp)
      builder.update(pt)
    }
    count.accumulateAndGet(1.0, sumOp)

    this
  }
}

object HistogramSummary {
  implicit val semigroup = new Semigroup[HistogramSummary] {
    def plus(l: HistogramSummary, r: HistogramSummary): HistogramSummary = l.merge(r)
  }
}
