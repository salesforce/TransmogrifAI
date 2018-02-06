/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.{FeatureBuilder => FB, FeatureBuilderWithExtract}
import com.salesforce.op.features.types._

/**
 * This file is currently used for Ensuring compatibility only;
 * once the functionality is in FeatureBuilder, it will be removed.
 */
object FeatureOps {
  def asPickList[T](f: T => Any): T => PickList = x => Option(f(x)).map(_.toString).toPickList
}
