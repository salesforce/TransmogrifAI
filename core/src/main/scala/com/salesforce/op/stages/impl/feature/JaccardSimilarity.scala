/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.utils.stats.JaccardDistance

/**
 * Calculates the Jaccard Similarity between two sets.
 * If both inputs are empty, Jaccard Similarity is defined as 1.0
 */
class JaccardSimilarity(uid: String = UID[JaccardSimilarity])
  extends BinaryTransformer[MultiPickList, MultiPickList, RealNN](operationName = "jacSim", uid = uid) {

  def transformFn: (MultiPickList, MultiPickList) => RealNN = (x, y) => JaccardDistance(x.value, y.value).toRealNN

}
