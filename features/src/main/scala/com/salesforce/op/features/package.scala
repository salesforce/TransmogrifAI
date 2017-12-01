/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op

import com.salesforce.op.features.types._


package object features {

  /**
   * Some common type shortcuts
   */
  type OPFeature = FeatureLike[_ <: FeatureType]

  type SingleResponseFeature = FeatureLike[_ <: SingleResponse]

}
