/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.hw.iris

import com.esotericsoftware.kryo.Kryo
import com.salesforce.op.utils.kryo.OpKryoRegistrator

class IrisKryoRegistrator extends OpKryoRegistrator {

  override def registerCustomClasses(kryo: Kryo): Unit = {
    doAvroRegistration[com.salesforce.hw.iris.Iris](kryo)
  }

}
