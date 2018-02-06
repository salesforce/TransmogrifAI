/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */


package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.{FeatureBuilder => FB}
import com.salesforce.op.cli.gen.templates.FeatureOps._

/**
 * This is a template for generating categorical feature handling in a generated project
 */
class CategoricalFeatureTemplate {
  private[templates] def feature =
  // BEGIN
  FB.PickList[SampleObject]
    .extract(asPickList(_.codeGeneration_categoricalField_codeGeneration))
  // END
}
