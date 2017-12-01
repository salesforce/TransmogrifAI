/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

/**
 * The name of the column containing the entity being scored will always be key
 */
case object DataFrameFieldNames {
  val KeyFieldName = "key"
  val RightKeyName = "rightKey"
  val CombinedKeyName = "combinedKey"
}
