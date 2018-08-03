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

package com.salesforce.op.aggregators

import com.salesforce.op.features.types._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives the union of Set values
 */
abstract class SetAggregator[A <: MultiPickList](implicit val ttag: WeakTypeTag[A])
  extends MonoidAggregator[Event[A], Set[String], A] with AggregatorDefaults[A] {
  val ftFactory = FeatureTypeFactory[A]()
  val monoid: Monoid[Set[String]] = Monoid.setMonoid[String]
}
case object UnionMultiPickList extends SetAggregator[MultiPickList]

/**
 * Set union semigroup
 */
private[op] class SetSemigroup[T] extends Semigroup[Set[T]] {

  final override def plus(left: Set[T], right: Set[T]): Set[T] =
    if (left.size > right.size) left ++ right else right ++ left

  final override def sumOption(items: TraversableOnce[Set[T]]): Option[Set[T]] =
    if (items.isEmpty) None
    else {
      val mutable = scala.collection.mutable.Set[T]()
      items.foreach { s => mutable ++= s }
      Some(mutable.toSet)
    }
}

