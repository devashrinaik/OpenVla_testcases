# Failure Mode Classification Report

## Methodology

Each failed episode is analyzed with 5 automatic detectors:
- **STUCK_LOOP**: Autocorrelation on late-episode actions detects cyclic repetition
- **GRASP_LOST**: Sustained gripper close followed by permanent open (had object, lost it)
- **AIMLESS_WANDERING**: High total movement but low net displacement (efficiency < 0.15)
- **GRIPPER_INDECISION**: Gripper switch rate > 6 per 100 steps
- **STALLED**: > 30% of steps with near-zero action magnitude

Episodes can have multiple labels (e.g., STUCK_LOOP + GRIPPER_INDECISION).

## Summary

Total failed episodes: 58

| Failure Mode | Count | % of Failed Episodes |
|---|---|---|
| GRIPPER_INDECISION | 26 | 45% |
| GRASP_LOST | 18 | 31% |
| STALLED | 16 | 28% |
| UNCLASSIFIED | 11 | 19% |
| STUCK_LOOP | 8 | 14% |
| AIMLESS_WANDERING | 7 | 12% |

## libero_spatial (25 failures)

- **Ep2** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_between_the_plate_and_the_r
- **Ep3** (220 steps) [STUCK_LOOP, GRASP_LOST, AIMLESS_WANDERING] — pick_up_the_black_bowl_between_the_plate_and_the_r
- **Ep0** (220 steps) [UNCLASSIFIED] — pick_up_the_black_bowl_from_table_center_and_place
- **Ep1** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_in_the_top_drawer_of_the_wo
- **Ep2** (220 steps) [UNCLASSIFIED] — pick_up_the_black_bowl_in_the_top_drawer_of_the_wo
- **Ep3** (220 steps) [UNCLASSIFIED] — pick_up_the_black_bowl_in_the_top_drawer_of_the_wo
- **Ep5** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_in_the_top_drawer_of_the_wo
- **Ep0** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep1** (220 steps) [GRASP_LOST] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep2** (220 steps) [GRASP_LOST] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep3** (220 steps) [STUCK_LOOP, GRASP_LOST, GRIPPER_INDECISION, STALLED] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep4** (220 steps) [GRASP_LOST, AIMLESS_WANDERING] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep5** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep6** (220 steps) [GRASP_LOST, AIMLESS_WANDERING] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep8** (220 steps) [GRASP_LOST] — pick_up_the_black_bowl_on_the_ramekin_and_place_it
- **Ep0** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_next_to_the_cookie_box_and_
- **Ep1** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_next_to_the_cookie_box_and_
- **Ep3** (220 steps) [STUCK_LOOP, GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_stove_and_place_it_o
- **Ep7** (220 steps) [GRASP_LOST, GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_stove_and_place_it_o
- **Ep9** (220 steps) [STALLED] — pick_up_the_black_bowl_on_the_stove_and_place_it_o
- **Ep2** (220 steps) [AIMLESS_WANDERING, GRIPPER_INDECISION] — pick_up_the_black_bowl_next_to_the_plate_and_place
- **Ep4** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_next_to_the_plate_and_place
- **Ep0** (220 steps) [STUCK_LOOP, GRASP_LOST, GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_wooden_cabinet_and_p
- **Ep5** (220 steps) [GRIPPER_INDECISION] — pick_up_the_black_bowl_on_the_wooden_cabinet_and_p
- **Ep8** (220 steps) [STUCK_LOOP, GRASP_LOST] — pick_up_the_black_bowl_on_the_wooden_cabinet_and_p

## libero_object (10 failures)

- **Ep1** (280 steps) [STUCK_LOOP, GRIPPER_INDECISION] — pick_up_the_alphabet_soup_and_place_it_in_the_bask
- **Ep2** (280 steps) [STALLED] — pick_up_the_alphabet_soup_and_place_it_in_the_bask
- **Ep4** (280 steps) [STALLED] — pick_up_the_alphabet_soup_and_place_it_in_the_bask
- **Ep7** (280 steps) [AIMLESS_WANDERING, GRIPPER_INDECISION] — pick_up_the_alphabet_soup_and_place_it_in_the_bask
- **Ep9** (280 steps) [STUCK_LOOP, GRIPPER_INDECISION] — pick_up_the_alphabet_soup_and_place_it_in_the_bask
- **Ep0** (280 steps) [GRASP_LOST, STALLED] — pick_up_the_cream_cheese_and_place_it_in_the_baske
- **Ep2** (280 steps) [STALLED] — pick_up_the_cream_cheese_and_place_it_in_the_baske
- **Ep5** (280 steps) [GRIPPER_INDECISION] — pick_up_the_cream_cheese_and_place_it_in_the_baske
- **Ep8** (280 steps) [GRASP_LOST, STALLED] — pick_up_the_cream_cheese_and_place_it_in_the_baske
- **Ep9** (280 steps) [AIMLESS_WANDERING, GRIPPER_INDECISION] — pick_up_the_salad_dressing_and_place_it_in_the_bas

## libero_goal (8 failures)

- **Ep3** (300 steps) [UNCLASSIFIED] — open_the_middle_drawer_of_the_cabinet
- **Ep7** (300 steps) [UNCLASSIFIED] — open_the_middle_drawer_of_the_cabinet
- **Ep0** (300 steps) [UNCLASSIFIED] — open_the_top_drawer_and_put_the_bowl_inside
- **Ep1** (300 steps) [UNCLASSIFIED] — open_the_top_drawer_and_put_the_bowl_inside
- **Ep2** (300 steps) [UNCLASSIFIED] — open_the_top_drawer_and_put_the_bowl_inside
- **Ep4** (300 steps) [UNCLASSIFIED] — open_the_top_drawer_and_put_the_bowl_inside
- **Ep6** (300 steps) [UNCLASSIFIED] — open_the_top_drawer_and_put_the_bowl_inside
- **Ep8** (300 steps) [GRIPPER_INDECISION] — open_the_top_drawer_and_put_the_bowl_inside

## libero_10 (15 failures)

- **Ep0** (520 steps) [GRIPPER_INDECISION] — LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_
- **Ep5** (520 steps) [UNCLASSIFIED] — LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_
- **Ep8** (520 steps) [GRASP_LOST, GRIPPER_INDECISION] — LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_
- **Ep0** (520 steps) [GRASP_LOST] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep1** (520 steps) [GRASP_LOST, GRIPPER_INDECISION] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep2** (520 steps) [STALLED] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep3** (520 steps) [GRIPPER_INDECISION, STALLED] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep4** (520 steps) [GRASP_LOST] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep5** (520 steps) [STALLED] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep6** (520 steps) [STALLED] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep8** (520 steps) [STALLED] — KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_dr
- **Ep0** (520 steps) [STUCK_LOOP, GRASP_LOST, STALLED] — LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_p
- **Ep1** (520 steps) [AIMLESS_WANDERING, GRIPPER_INDECISION, STALLED] — LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_p
- **Ep2** (520 steps) [STALLED] — LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_p
- **Ep6** (520 steps) [GRASP_LOST, GRIPPER_INDECISION, STALLED] — LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_p

## Implications for External Memory Module

- **8 STUCK_LOOP failures**: Memory should detect action repetition and trigger re-planning or exploration
- **18 GRASP_LOST failures**: Memory should track grasp state and trigger recovery sub-routine
- **7 AIMLESS_WANDERING failures**: Memory should maintain goal representation to prevent drift
- **26 GRIPPER_INDECISION failures**: Memory should commit to grasp/release decisions (reduce oscillation)
- **16 STALLED failures**: Memory should detect inactivity and inject exploratory actions