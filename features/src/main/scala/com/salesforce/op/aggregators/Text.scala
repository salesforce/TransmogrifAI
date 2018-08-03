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
import com.salesforce.op.utils.text.TextUtils
import com.salesforce.op.utils.tuples.RichTuple._
import com.twitter.algebird._

import scala.reflect.runtime.universe._

/**
 * Aggregator that gives separated concatenation of text values with a separator
 */
abstract class ConcatTextWithSeparator[T <: Text]
(
  val separator: String = ","
)(implicit val ttag: WeakTypeTag[T])
  extends MonoidAggregator[Event[T], Option[String], T]
    with AggregatorDefaults[T] {
  val ftFactory = FeatureTypeFactory[T]()
  val monoid: Monoid[Option[String]] = new Monoid[Option[String]] {
    val zero: Option[String] = None
    def plus(l: Option[String], r: Option[String]): Option[String] =
      (l -> r).map(TextUtils.concat(_, _, separator = separator))
  }
}
case object ConcatBase64 extends ConcatTextWithSeparator[Base64]
case object ConcatComboBox extends ConcatTextWithSeparator[ComboBox]
case object ConcatEmail extends ConcatTextWithSeparator[Email]
case object ConcatID extends ConcatTextWithSeparator[ID]
case object ConcatPhone extends ConcatTextWithSeparator[Phone]
case object ConcatText extends ConcatTextWithSeparator[Text](separator = " ")
case object ConcatTextArea extends ConcatTextWithSeparator[TextArea](separator = " ")
case object ConcatURL extends ConcatTextWithSeparator[URL]
case object ConcatCountry extends ConcatTextWithSeparator[Country]
case object ConcatState extends ConcatTextWithSeparator[State]
case object ConcatCity extends ConcatTextWithSeparator[City]
case object ConcatPostalCode extends ConcatTextWithSeparator[PostalCode]
case object ConcatStreet extends ConcatTextWithSeparator[Street]

/**
 * Aggregator for PickLists that gives the most common non-Empty value seen during the aggregation
 */
case object ModePickList extends MonoidAggregator[Event[PickList], Map[String, Int], PickList] {
  override def prepare(input: Event[PickList]): Map[String, Int] =
    input.value.map(x => Map(x -> 1)).getOrElse(Map.empty[String, Int])
  override def present(reduction: Map[String, Int]): PickList = {
    // Only return an empty PickList if the all the PickLists in the aggregation are empty
    if (reduction.isEmpty) PickList.empty else reduction.minBy(x => (-x._2, x._1))._1.toPickList
  }
  val monoid: Monoid[Map[String, Int]] = Monoid.mapMonoid[String, Int]
}
