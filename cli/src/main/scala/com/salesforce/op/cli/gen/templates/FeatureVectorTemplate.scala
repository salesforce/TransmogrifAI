/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen.templates

import com.salesforce.op.features.Feature
import com.salesforce.op.features.types._

import com.salesforce.op._
import com.salesforce.op.features.types._

/**
 * This is a template for generating vector feature transformation in a generated project
 */
class FeatureVectorTemplate {
  val codeGeneration_list_codeGeneration: Seq[Feature[Binary]] = List(new BinaryFeatureTemplate().feature.asPredictor)
  // BEGIN
  codeGeneration_list_codeGeneration.transmogrify()
  // END
}
