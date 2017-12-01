/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
    input.value.value.map(x => Map(x -> 1)).getOrElse(Map.empty[String, Int])
  override def present(reduction: Map[String, Int]): PickList = {
    // Only return an empty PickList if the all the PickLists in the aggregation are empty
    if (reduction.isEmpty) PickList.empty
    else {
      val res: String = reduction.toSeq.sortBy(x => (-x._2, x._1)).head._1
      PickList(res)
    }
  }
  val monoid: Monoid[Map[String, Int]] = Monoid.mapMonoid[String, Int]
}
