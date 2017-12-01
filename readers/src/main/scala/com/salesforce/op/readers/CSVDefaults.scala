/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.readers

import com.salesforce.op.utils.date.DateTimeUtils
import com.salesforce.op.utils.io.csv.CSVOptions

object CSVDefaults {
  val CSVOptions: CSVOptions = new CSVOptions()
  val TimeZone: String = DateTimeUtils.DefaultTimeZoneStr
  val RecordNamespace: String = "salesforce"
  val RecordName: String = "AutoInferredRecord"
}
