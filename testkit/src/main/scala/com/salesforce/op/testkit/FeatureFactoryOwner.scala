/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types.{FeatureType, FeatureTypeFactory}

import scala.reflect.runtime.universe._

private[testkit] abstract class FeatureFactoryOwner[FT <: FeatureType : WeakTypeTag] {
  protected val ftFactory: FeatureTypeFactory[FT] = FeatureTypeFactory[FT]()
}
