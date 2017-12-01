/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.unary.UnaryTransformer

import scala.reflect.runtime.universe.TypeTag

/** Filters maps by keys provided in a whiteList or blackList
 *
 * @param uid uid for instance
 * @param tti type tag for input
 * @tparam I input feature type
 */
class FilterMap[I <: OPMap[_]](uid: String = UID[FilterMap[_]])(implicit tti: TypeTag[I])
  extends UnaryTransformer[I, I](operationName = "filterMap", uid = uid)
    with MapPivotParams with TextParams with CleanTextMapFun {

  private val ftFactory = FeatureTypeFactory[I]()

  override def transformFn: I => I = (origMap: I) => {
    val filtered = filterKeys[Any](origMap.v, shouldCleanKey = $(cleanKeys), shouldCleanValue = $(cleanText))
    ftFactory.newInstance(filtered)
  }
}
