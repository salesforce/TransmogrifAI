/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer

import scala.reflect.runtime.universe.TypeTag

object ToOccurTransformer {
  private def defaultMatches[T <: FeatureType](value: T): Boolean = value match {
    case num: OPNumeric[_] if num.nonEmpty => num.toDouble.get > 0.0
    case text: Text if text.nonEmpty => text.value.get.length > 0
    case collection: OPCollection => collection.nonEmpty
    case _ => false
  }
}

/**
 * Transformer that converts input feature of type I into doolean feature using a user specified function that
 * maps object type I to a Boolean
 *
 * @param uid     uid for instance
 * @param matchFn A function that allows the user to pass in a function that maps object I to a Boolean.
 * @tparam I Object type to be mapped to a double (doolean).
 */
class ToOccurTransformer[I <: FeatureType]
(
  uid: String = UID[ToOccurTransformer[I]],
  val matchFn: I => Boolean = ToOccurTransformer.defaultMatches[I] _
)(implicit tti: TypeTag[I])
  extends UnaryTransformer[I, RealNN](operationName = "toOccur", uid = uid) {

  private val (yes, no) = (RealNN(1.0), RealNN(0.0))

  def transformFn: I => RealNN = (value: I) => if (matchFn(value)) yes else no

}
