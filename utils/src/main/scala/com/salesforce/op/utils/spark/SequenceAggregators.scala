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

package com.salesforce.op.utils.spark

import com.twitter.algebird.Operators._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{Encoder, Encoders}

/**
 * A factory for Spark sequence aggregators
 */
object SequenceAggregators {

  // scalastyle:off method.name

  // TODO: try using Algebird monoid to avoid code repetition below

  /**
   * Creates aggregator that sums a Dataset column of type Seq[T: Numeric]
   *
   * @param size the size of the Sequence
   * @tparam T numeric type
   * @return spark aggregator
   */
  def SumNumSeq[T: Numeric : Encoder](size: Int)(implicit enc: Encoder[Seq[T]]): Aggregator[Seq[T], Seq[T], Seq[T]] = {
    val numeric = implicitly[Numeric[T]]
    new Aggregator[Seq[T], Seq[T], Seq[T]] {
      val zero: Seq[T] = Seq.fill(size)(numeric.zero)
      def reduce(b: Seq[T], a: Seq[T]): Seq[T] = b.zip(a).map { case (m1, m2) => numeric.plus(m1, m2) }
      def merge(b: Seq[T], a: Seq[T]): Seq[T] = reduce(b, a)
      def finish(reduction: Seq[T]): Seq[T] = reduction
      def outputEncoder: Encoder[Seq[T]] = enc
      def bufferEncoder: Encoder[Seq[T]] = enc
    }
  }

  type SeqD = Seq[Double]
  type SeqOptD = Seq[Option[Double]]
  type SeqTupD = Seq[(Double, Int)]

  /**
   * Creates aggregator that computes mean on a Dataset column of type Seq[Option[Double]]
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def MeanSeqNullNum(size: Int): Aggregator[SeqOptD, SeqTupD, SeqD] = {
    new Aggregator[SeqOptD, SeqTupD, SeqD] {
      val zero: SeqTupD = Seq.fill(size)((0.0, 0))
      def reduce(b: SeqTupD, a: SeqOptD): SeqTupD = b.zip(a).map {
        case ((s, c), Some(v)) => (s + v, c + 1)
        case (sc, None) => sc
      }
      def merge(b1: SeqTupD, b2: SeqTupD): SeqTupD = b1.zip(b2).map { case (cs1, cs2) => cs1 + cs2 }
      def finish(reduction: SeqTupD): SeqD = reduction.map { case (s, c) => if (c > 0) s / c else s }
      def bufferEncoder: Encoder[SeqTupD] = ExpressionEncoder()
      def outputEncoder: Encoder[SeqD] = ExpressionEncoder()
    }
  }

  type SeqL = Seq[Long]
  type SeqOptL = Seq[Option[Long]]
  type SeqMapLL = Seq[Map[Long, Long]]

  /**
   * Creates aggregator that computes mode on a Dataset column of type Seq[Option[Long]]
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def ModeSeqNullInt(size: Int): Aggregator[SeqOptL, SeqMapLL, SeqL] = {
    new Aggregator[SeqOptL, SeqMapLL, SeqL] {
      // Here, empty maps correspond to a zero element, this is the initial value for the reduce
      val zero: SeqMapLL = Seq.fill(size)(Map.empty)

      // At the reduce step, we keep track of how many times we've seen an element by adding/updating an
      // element -> count entry to our map
      def reduce(b: SeqMapLL, a: SeqOptL): SeqMapLL = b.zip(a).map {
        // Increment counter for new value if it's already in the map, otherwise add it to the map
        case (m, Some(newV)) =>
          val counter = m.get(newV).map(_ + 1L).getOrElse(1L)
          m + (newV -> counter)

        // Do nothing if the new element to add is None
        case (m, None) => m
      }

      // At the merge step combine the multiple map sequences into a single map sequence, incrementing the
      // individual counters as necessary (just use Algebird's + operator!)
      def merge(ms1: SeqMapLL, ms2: SeqMapLL): SeqMapLL = ms1.zip(ms2).map { case (m1, m2) => m1 + m2 }

      // At the finish step, we sort the final maps by value and take the corresponding key to find the mode
      def finish(reduction: SeqMapLL): SeqL = {
        // Sort by negative count and positive value to get a high-low sort by counts sorted low-high by value,
        // (all counts are >= 1L). Note that we are not guaranteed that each map is nonEmpty so be sure to check
        // for that as well. Taking the head should give the mode, where ties are broken by choosing the smallest value
        reduction.map(m => if (m.isEmpty) 0L else m.minBy(x => (-x._2, x._1))._1)
      }
      def bufferEncoder: Encoder[SeqMapLL] = ExpressionEncoder()
      def outputEncoder: Encoder[SeqL] = ExpressionEncoder()
    }
  }

  type MapMap = Map[String, Map[String, Long]]
  type SeqMapMap = Seq[MapMap]

  /**
   * Creates aggregator that sums a Dataset column of type Seq[Map[String, Map[String, Long]]] such that the maps are
   * summed with the keys preserved and the values resulting are the sum of the values for the two maps
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def SumSeqMapMap(size: Int): Aggregator[SeqMapMap, SeqMapMap, SeqMapMap] = {
    new Aggregator[SeqMapMap, SeqMapMap, SeqMapMap] {
      val zero: SeqMapMap = Seq.fill(size)(Map.empty)
      def reduce(b: SeqMapMap, a: SeqMapMap): SeqMapMap = b.zip(a).map { case (m1, m2) => m1 + m2 }
      def merge(b: SeqMapMap, a: SeqMapMap): SeqMapMap = reduce(b, a)
      def finish(reduction: SeqMapMap): SeqMapMap = reduction
      def bufferEncoder: Encoder[SeqMapMap] = Encoders.kryo[SeqMapMap]
      def outputEncoder: Encoder[SeqMapMap] = Encoders.kryo[SeqMapMap]
    }
  }

  type SeqSet = Seq[Set[String]]

  /**
   * Creates aggregator that sums a Dataset column of type Seq[Set[String]]
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def SumSeqSet(size: Int): Aggregator[SeqSet, SeqSet, SeqSet] = {
    new Aggregator[SeqSet, SeqSet, SeqSet] {
      val zero: SeqSet = Seq.fill(size)(Set.empty)
      def reduce(b: SeqSet, a: SeqSet): SeqSet = b.zip(a).map { case (m1, m2) => m1 + m2 }
      def merge(b: SeqSet, a: SeqSet): SeqSet = reduce(b, a)
      def finish(reduction: SeqSet): SeqSet = reduction
      def bufferEncoder: Encoder[SeqSet] = Encoders.kryo[SeqSet]
      def outputEncoder: Encoder[SeqSet] = Encoders.kryo[SeqSet]
    }
  }

  type SeqMapDouble = Seq[Map[String, Double]]
  type SeqMapTuple = Seq[Map[String, (Double, Int)]]

  /**
   * Creates aggregator that computes the means by key of a Dataset column of type Seq[Map[String, Double]].
   * Each map has a separate mean by key computed.
   * Because each map does not have to have all the possible keys,
   * the element counts for each map's keys can all be different.
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def MeanSeqMapDouble(size: Int): Aggregator[SeqMapDouble, SeqMapTuple, SeqMapDouble] = {
    new Aggregator[SeqMapDouble, SeqMapTuple, SeqMapDouble] {
      val zero: SeqMapTuple = Seq.fill(size)(Map.empty)
      def reduce(b: SeqMapTuple, a: SeqMapDouble): SeqMapTuple =
        merge(b, a.map(_.map { case (k, v) => k -> (v, 1) }))
      def merge(b1: SeqMapTuple, b2: SeqMapTuple): SeqMapTuple = b1.zip(b2).map { case (m1, m2) => m1 + m2 }
      def finish(r: SeqMapTuple): SeqMapDouble = r.map(m =>
        m.map { case (k, (s, c)) => (k, if (c > 0) s / c else s) }
      )
      // Seq of Map of Tuple is too complicated for Spark's encoder, so need to use Kryo's
      def bufferEncoder: Encoder[SeqMapTuple] = Encoders.kryo[SeqMapTuple]
      def outputEncoder: Encoder[SeqMapDouble] = ExpressionEncoder()
    }
  }

  type SeqMapLong = Seq[Map[String, Long]]
  type SeqMapMapLong = Seq[Map[String, Map[Long, Long]]]

  /**
   * Creates aggregator that computes the modes by key of a Dataset column of type Seq[Map[String, Long]].
   * Each map has a separate mode by key computed.
   * Because each map does not have to have all the possible keys,
   * the element counts for each map's keys can all be different.
   *
   * @param size the size of the Sequence
   * @return spark aggregator
   */
  def ModeSeqMapLong(size: Int): Aggregator[SeqMapLong, SeqMapMapLong, SeqMapLong] = {
    new Aggregator[SeqMapLong, SeqMapMapLong, SeqMapLong] {
      val zero: SeqMapMapLong = Seq.fill(size)(Map.empty)
      def reduce(b: SeqMapMapLong, a: SeqMapLong): SeqMapMapLong =
        merge(b, a.map(_.map { case (k, v) => k -> Map(v -> 1L) }.toMap))
      def merge(b1: SeqMapMapLong, b2: SeqMapMapLong): SeqMapMapLong = b1.zip(b2).map { case (m1, m2) => m1 + m2 }
      def finish(r: SeqMapMapLong): SeqMapLong = r.map(_.map {
        case (k, m) => k -> (if (m.isEmpty) 0L else m.minBy(x => (-x._2, x._1))._1) }
      )
      def bufferEncoder: Encoder[SeqMapMapLong] = ExpressionEncoder()
      def outputEncoder: Encoder[SeqMapLong] = ExpressionEncoder()
    }
  }

  type SeqMapMapInt = Seq[Map[Int, Map[String, Int]]]

  def SumSeqMapMapInt(size: Int): Aggregator[SeqMapMapInt, SeqMapMapInt, SeqMapMapInt] = {
    new Aggregator[SeqMapMapInt, SeqMapMapInt, SeqMapMapInt] {
      val zero: SeqMapMapInt = Seq.fill(size)(Map.empty)
      def reduce(b: SeqMapMapInt, a: SeqMapMapInt): SeqMapMapInt = b.zip(a).map { case (m1, m2) => m1 + m2 }
      def merge(b: SeqMapMapInt, a: SeqMapMapInt): SeqMapMapInt = reduce(b, a)
      def finish(reduction: SeqMapMapInt): SeqMapMapInt = reduction
      def bufferEncoder: Encoder[SeqMapMapInt] = Encoders.kryo[SeqMapMapInt]
      def outputEncoder: Encoder[SeqMapMapInt] = Encoders.kryo[SeqMapMapInt]
    }
  }

  type SeqMapMapMapInt = Seq[Map[String, Map[Int, Map[String, Int]]]]

  def SumSeqMapMapMapInt(size: Int): Aggregator[SeqMapMapMapInt, SeqMapMapMapInt, SeqMapMapMapInt] = {
    new Aggregator[SeqMapMapMapInt, SeqMapMapMapInt, SeqMapMapMapInt] {
      val zero: SeqMapMapMapInt = Seq.fill(size)(Map.empty)
      def reduce(b: SeqMapMapMapInt, a: SeqMapMapMapInt): SeqMapMapMapInt = b.zip(a).map { case (m1, m2) => m1 + m2 }
      def merge(b: SeqMapMapMapInt, a: SeqMapMapMapInt): SeqMapMapMapInt = reduce(b, a)
      def finish(reduction: SeqMapMapMapInt): SeqMapMapMapInt = reduction
      def bufferEncoder: Encoder[SeqMapMapMapInt] = Encoders.kryo[SeqMapMapMapInt]
      def outputEncoder: Encoder[SeqMapMapMapInt] = Encoders.kryo[SeqMapMapMapInt]
    }
  }

}
