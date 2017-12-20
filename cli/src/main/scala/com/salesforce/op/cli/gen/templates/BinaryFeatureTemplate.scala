/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

/**
 * This is a template for generating binary feature handling in a generated project
 */
class BinaryFeatureTemplate {
  private[templates] def feature =
  // BEGIN
  FeatureBuilder.Binary[SampleObject].
    extract(o => Option(o.codeGeneration_binaryField_codeGeneration).map(_.booleanValue).toBinary)
  // END
}
