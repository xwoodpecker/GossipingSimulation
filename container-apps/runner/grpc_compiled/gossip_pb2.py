# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gossip.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cgossip.proto\x12\x06gossip\"\x0f\n\rGossipRequest\"\x10\n\x0eGossipResponse\"\x10\n\x0eHistoryRequest\"<\n\x0fHistoryResponse\x12)\n\rvalue_entries\x18\x01 \x03(\x0b\x32\x12.gossip.ValueEntry\"3\n\nValueEntry\x12\x16\n\x0eparticipations\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05\"\x15\n\x13\x43urrentValueRequest\"%\n\x14\x43urrentValueResponse\x12\r\n\x05value\x18\x01 \x01(\x05\"\x18\n\x16StopApplicationRequest\"\x19\n\x17StopApplicationResponse2\xa4\x02\n\x06Gossip\x12\x39\n\x06Gossip\x12\x15.gossip.GossipRequest\x1a\x16.gossip.GossipResponse\"\x00\x12<\n\x07History\x12\x16.gossip.HistoryRequest\x1a\x17.gossip.HistoryResponse\"\x00\x12K\n\x0c\x43urrentValue\x12\x1b.gossip.CurrentValueRequest\x1a\x1c.gossip.CurrentValueResponse\"\x00\x12T\n\x0fStopApplication\x12\x1e.gossip.StopApplicationRequest\x1a\x1f.gossip.StopApplicationResponse\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'gossip_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GOSSIPREQUEST._serialized_start=24
  _GOSSIPREQUEST._serialized_end=39
  _GOSSIPRESPONSE._serialized_start=41
  _GOSSIPRESPONSE._serialized_end=57
  _HISTORYREQUEST._serialized_start=59
  _HISTORYREQUEST._serialized_end=75
  _HISTORYRESPONSE._serialized_start=77
  _HISTORYRESPONSE._serialized_end=137
  _VALUEENTRY._serialized_start=139
  _VALUEENTRY._serialized_end=190
  _CURRENTVALUEREQUEST._serialized_start=192
  _CURRENTVALUEREQUEST._serialized_end=213
  _CURRENTVALUERESPONSE._serialized_start=215
  _CURRENTVALUERESPONSE._serialized_end=252
  _STOPAPPLICATIONREQUEST._serialized_start=254
  _STOPAPPLICATIONREQUEST._serialized_end=278
  _STOPAPPLICATIONRESPONSE._serialized_start=280
  _STOPAPPLICATIONRESPONSE._serialized_end=305
  _GOSSIP._serialized_start=308
  _GOSSIP._serialized_end=600
# @@protoc_insertion_point(module_scope)
