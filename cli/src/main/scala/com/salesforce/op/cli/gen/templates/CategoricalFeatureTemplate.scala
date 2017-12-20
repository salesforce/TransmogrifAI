/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._

/**
 * This is a template for generating categorical feature handling in a generated project
 */
class CategoricalFeatureTemplate {
  private[templates] def feature =
  // BEGIN
  FeatureBuilder.MultiPickList[SampleObject]
    .extract(o => Option(o.codeGeneration_categoricalField_codeGeneration)
    .map(_.toString).toSet[String].toMultiPickList)
  // END
}
