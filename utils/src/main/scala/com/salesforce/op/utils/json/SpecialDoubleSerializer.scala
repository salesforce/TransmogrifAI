/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.json

import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

/**
 * Json4s serializer for marshalling special Double values: NaN, -Infinity and Infinity
 */
// scalastyle:off
class SpecialDoubleSerializer extends CustomSerializer[Double](_ =>
  ({
    case JString("NaN") => Double.NaN
    case JString("-Infinity") => Double.NegativeInfinity
    case JString("Infinity") => Double.PositiveInfinity
  }, {
    case v: Double if v.isNaN => JString("NaN")
    case Double.NegativeInfinity => JString("-Infinity")
    case Double.PositiveInfinity => JString("Infinity")
  }))
