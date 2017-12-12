package com.salesforce.hw.boston

import com.esotericsoftware.kryo.Kryo
import com.salesforce.op.utils.kryo.OpKryoRegistrator


class BostonKryoRegistrator extends OpKryoRegistrator {

  override def registerCustomClasses(kryo: Kryo): Unit = {
    doClassRegistration(kryo)(classOf[BostonFeatures])
    doClassRegistration(kryo)(classOf[BostonHouse])
  }

}
