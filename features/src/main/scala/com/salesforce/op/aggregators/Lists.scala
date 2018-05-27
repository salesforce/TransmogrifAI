/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives concatenation of the lists
 */
abstract class ConcatList[V, T <: OPList[V]](implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Seq[V], T]
    with AggregatorDefaults[T] {
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()
  val monoid: Monoid[Seq[V]] = Monoid.seqMonoid[V]

}
case object ConcatTextList extends ConcatList[String, TextList]
case object ConcatDateList extends ConcatList[Long, DateList]
case object ConcatDateTimeList extends ConcatList[Long, DateTimeList]


/**
 * Aggregator that gives min/max item over a collection of lists
 */
abstract class MinMaxList[N, T <: OPList[N]]
(
  isMin: Boolean
)(implicit val ord: Ordering[N], ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Seq[N], T] {
  val ftFactory: FeatureTypeFactory[T] = FeatureTypeFactory[T]()

  def prepare(input: Event[T]): Seq[N] = prepareFn(input.value.value)
  def present(reduction: Seq[N]): T = ftFactory.newInstance(prepareFn(reduction))
  private def prepareFn(v: Seq[N]): Seq[N] = if (v.isEmpty) Seq.empty else Seq(v.reduce(ordFn))

  val monoid: Monoid[Seq[N]] = Monoid.seqMonoid[N]
  val ordFn: (N, N) => N = if (isMin) ord.min _ else ord.max _

}
case object MaxDateList extends MinMaxList[Long, DateList](isMin = false)
case object MaxDateTimeList extends MinMaxList[Long, DateTimeList](isMin = false)
case object MinDateList extends MinMaxList[Long, DateList](isMin = true)
case object MinDateTimeList extends MinMaxList[Long, DateTimeList](isMin = true)
