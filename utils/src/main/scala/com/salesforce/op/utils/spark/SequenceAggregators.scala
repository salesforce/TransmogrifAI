/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import com.twitter.algebird.Operators._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{Encoder, TypedColumn}

import scala.Numeric.Implicits._


object SequenceAggregators {

  // scalastyle:off method.name

  // TODO: try using Algebird monoid to avoid code repetition below
  /**
   * Creates a TypedColumn that sums a Dataset column of type Seq[T: Numeric]
   *
   * @param size The size of the Sequence
   * @tparam T The Numeric type
   * @return a TypedColumn that can be used in Dataset
   */
  def SumNumSeq[T: Numeric : Encoder](size: Int)(implicit enc: Encoder[Seq[T]]): TypedColumn[Seq[T], Seq[T]] = {
    val numeric = implicitly[Numeric[T]]
    new Aggregator[Seq[T], Seq[T], Seq[T]] {
      override def zero: Seq[T] = Seq.fill(size)(numeric.zero)
      override def reduce(b: Seq[T], a: Seq[T]): Seq[T] = b.zip(a).map { case (m1, m2) => m1 + m2 }
      override def merge(b: Seq[T], a: Seq[T]): Seq[T] = reduce(b, a)
      override def finish(reduction: Seq[T]): Seq[T] = reduction
      override def outputEncoder: Encoder[Seq[T]] = enc
      override def bufferEncoder: Encoder[Seq[T]] = enc
    }.toColumn
  }

  type SeqD = Seq[Double]
  type SeqOptD = Seq[Option[Double]]
  type SeqTupD = Seq[(Double, Double)]

  /**
   * Creates a TypedColumn that computes mean on a Dataset column of type Seq[Option[Double]]
   *
   * @param size The size of the Sequence
   * @return a TypedColumn that can be used in Dataset
   */
  def MeanSeqNullNum(size: Int): TypedColumn[SeqOptD, SeqD] = {
    new Aggregator[SeqOptD, SeqTupD, SeqD] {
      override def zero: SeqTupD = Seq.fill(size)((0.0, 0.0))
      override def reduce(b: SeqTupD, a: SeqOptD): SeqTupD =
        b.zip(a).map {
          case ((c, s), Some(v)) => (c + 1.0, s + v)
          case ((c, s), None) => (c, s)
        }
      override def merge(b1: SeqTupD, b2: SeqTupD): SeqTupD = b1.zip(b2).map { case (cs1, cs2) => cs1 + cs2 }
      override def finish(reduction: SeqTupD): SeqD =
        reduction.map {
          case (c, s) if c != 0.0 => s / c
          case (c, s) if c == 0.0 => 0.0
        }
      override def bufferEncoder: Encoder[SeqTupD] = ExpressionEncoder()
      override def outputEncoder: Encoder[SeqD] = ExpressionEncoder()
    }.toColumn
  }

  type SeqL = Seq[Long]
  type SeqOptL = Seq[Option[Long]]
  type SeqMapLL = Seq[Map[Long, Long]]

  /**
   * Creates a TypedColumn that computes mode on a Dataset column of type Seq[Option[Long]]
   *
   * @param size The size of the Sequence
   * @return a TypedColumn that can be used in Dataset
   */
  def ModeSeqNullInt(size: Int): TypedColumn[SeqOptL, SeqL] = {
    new Aggregator[SeqOptL, SeqMapLL, SeqL] {
      // Here, empty maps correspond to a zero element, this is the initial value for the reduce
      override def zero: SeqMapLL = Seq.fill(size)(Map[Long, Long]())
      // At the reduce step, we keep track of how many times we've seen an element by adding/updating an
      // element -> count entry to our map
      override def reduce(b: SeqMapLL, a: SeqOptL): SeqMapLL =
      b.zip(a).map {
        // Increment counter for new value if it's already in the map, otherwise add it to the map
        case (m, Some(newV)) => {
          val counter = m.get(newV).map(_ + 1L).getOrElse(1L)
          m + (newV -> counter)
        }
        // Do nothing if the new element to add is None
        case (m, None) => m
      }
      // At the merge step combine the multiple map sequences into a single map sequence, incrementing the
      // individual counters as necessary (just use Algebird's + operator!)
      override def merge(ms1: SeqMapLL, ms2: SeqMapLL): SeqMapLL = ms1.zip(ms2).map { case (m1, m2) => m1 + m2 }
      // At the finish step, we sort the final maps by value and take the corresponding key to find the mode
      override def finish(reduction: SeqMapLL): SeqL = {
        // Sort by negative count and positive value to get a high-low sort by counts sorted low-high by value,
        // (all counts are >= 1L). Note that we are not guaranteed that each map is nonEmpty so be sure to check
        // for that as well. Taking the head should give the mode, where ties are broken by choosing the smallest value
        reduction.map(m => m.toSeq.sortBy(r => (-r._2, r._1)).headOption.getOrElse((0L, 0L))._1)
      }
      override def bufferEncoder: Encoder[SeqMapLL] = ExpressionEncoder()
      override def outputEncoder: Encoder[SeqL] = ExpressionEncoder()
    }.toColumn
  }

  type MapMap = Map[String, Map[String, Long]]
  type SeqMapMap = Seq[MapMap]

  /**
   * Creates a TypedColumn that sums a Dataset column of type Seq[Map[String, Map[String, Long]]] such that the maps are
   * summed with the keys preserved and the values resulting are the sum of the values for the two maps
   *
   * @param size The size of the Sequence
   * @return a TypedColumn that can be used in Dataset
   */
  def SumSeqMapMap(size: Int)(implicit enc: Encoder[SeqMapMap]): TypedColumn[SeqMapMap, SeqMapMap] = {
    new Aggregator[SeqMapMap, SeqMapMap, SeqMapMap] {
      override def zero: SeqMapMap = Seq.fill(size)(Map[String, Map[String, Long]]())
      override def reduce(b: SeqMapMap, a: SeqMapMap): SeqMapMap = b.zip(a).map { case (m1, m2) => m1 + m2 }
      override def merge(b: SeqMapMap, a: SeqMapMap): SeqMapMap = reduce(b, a)
      override def finish(reduction: SeqMapMap): SeqMapMap = reduction
      override def bufferEncoder: Encoder[SeqMapMap] = enc
      override def outputEncoder: Encoder[SeqMapMap] = enc
    }.toColumn
  }


  type SeqSet = Seq[Set[String]]

  /**
   * Creates a TypedColumn that sums a Dataset column of type Seq[Set[String]] such that the maps are
   * summed with the keys preserved and the values resulting are the sum of the values for the two maps
   *
   * @param size The size of the Sequence
   * @return a TypedColumn that can be used in Dataset
   */
  def SumSeqSet(size: Int)(implicit enc: Encoder[SeqSet]): TypedColumn[SeqSet, SeqSet] = {
    new Aggregator[SeqSet, SeqSet, SeqSet] {
      override def zero: SeqSet = Seq.fill(size)(Set[String]())
      override def reduce(b: SeqSet, a: SeqSet): SeqSet = b.zip(a).map { case (m1, m2) => m1 + m2 }
      override def merge(b: SeqSet, a: SeqSet): SeqSet = reduce(b, a)
      override def finish(reduction: SeqSet): SeqSet = reduction
      override def bufferEncoder: Encoder[SeqSet] = enc
      override def outputEncoder: Encoder[SeqSet] = enc
    }.toColumn
  }
}
