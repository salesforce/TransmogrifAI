/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

/**
 * This is a template for generating real feature handling in a generated project
 */
class RealFeatureTemplate {
  private[templates] def feature =
  // BEGIN
  FeatureBuilder.Real[SampleObject].extract(o => o.codeGeneration_realField_codeGeneration.toReal)
  // END
}
