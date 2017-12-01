/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features

class FeatureCycleException(from: OPFeature, to: OPFeature)
  extends Exception("Cycle detected at " + to + " while traversing from " + from)
