# Postmortem: checkout outage on 12 February

Status: final. Blameless format.

## Summary

For 47 minutes, between 14:05 and 14:52, around 60% of checkout requests failed with a server error. Customers could browse and fill their baskets, but most could not complete a purchase. Roughly 1,900 orders failed; some customers retried successfully, and support received about 130 complaints.

## Timeline

- 13:58 deploy of release 4.18 starts
- 14:05 error rate on the order service starts climbing
- 14:19 first alert fires (error rate above 5% for 10 minutes), on-call engineer acknowledges
- 14:31 the deploy is identified as the likely cause, rollback started
- 14:44 rollback complete, but errors continue because existing connections are stuck
- 14:52 order service pods restarted, error rate back to normal

## Root cause

Release 4.18 changed the database client configuration and accidentally dropped the connection timeout setting during a refactor. Under normal load, slow queries now held on to their connections indefinitely, and the pool of 50 connections ran dry within minutes. Every new checkout request then waited for a free connection and timed out.

## What went well

- the rollback procedure worked as written
- support had a status page message up within 15 minutes

## What went badly

- it took 14 minutes for an alert to fire, and the alert said "high error rate" rather than pointing at the database pool
- the rollback alone did not fix it, which cost another 8 minutes of confusion

## Action items

1. Add an alert on connection pool saturation (owner: Ben, due 1 March).
2. Add the timeout to the config schema with a sane default so it cannot be dropped silently (owner: me).
3. Load test the order service before each release that touches the database client (owner: Marta).
4. Update the rollback runbook: restart the pods after rolling back database client changes.
