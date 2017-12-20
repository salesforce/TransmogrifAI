/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

/**
 * This is a template for generating text feature handling in a generated project
 */
class TextFeatureTemplate {
  private[templates] def feature =
  // BEGIN
  FeatureBuilder.Text[SampleObject].extract(o => o.codeGeneration_textField_codeGeneration.toText)
  // END
}
