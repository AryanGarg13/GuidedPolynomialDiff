import pybullet_planning as pp


# pp.interfaces.planner_interface.joint_motion_planning.get_sample_fn(env.manipulator,env.joints,)


def get_sample_fn(env, **kwargs):
    lower_limits = env.joint_lower_limits
    upper_limits = env.joint_upper_limits
    generator = pp.interfaces.planner_interface.joint_motion_planning.interval_generator(lower_limits, upper_limits, **kwargs)
    def fn():
        return tuple(next(generator))
    return fn

def get_collision_fn(body, joints,env, obstacles=[],
                    attachments=[], self_collisions=True,
                    disabled_collisions={},
                    extra_disabled_collisions={},
                    body_name_from_id=None, **kwargs):
    from pybullet_planning.interfaces.env_manager.pose_transformation import all_between
    from pybullet_planning.interfaces.robots.joint import set_joint_positions, get_custom_limits
    from pybullet_planning.interfaces.robots.link import get_self_link_pairs, get_moving_links
    from pybullet_planning.interfaces.debug_utils.debug_utils import draw_collision_diagnosis
    moving_links = frozenset(get_moving_links(body, joints))
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [(body, moving_links)] + attached_bodies
    # * main body self-collision link pairs
    self_check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
    # * main body link - attachment body pairs
    attach_check_pairs = []
    for attached in attachments:
        if attached.parent != body:
            continue
        # prune the main body link adjacent to the attachment and the ones in ignored collisions
        # TODO: prune the link that's adjacent to the attach link as well?
        # i.e. object attached to ee_tool_link, and ee geometry is attached to ee_base_link
        # TODO add attached object's link might not be BASE_LINK (i.e. actuated tool)
        # get_all_links
        at_check_links = []
        for ml in moving_links:
            if ml != attached.parent_link and \
                ((body, ml), (attached.child, pp.interfaces.robots.BASE_LINK)) not in extra_disabled_collisions and \
                ((attached.child, pp.interfaces.robots.BASE_LINK), (body, ml)) not in extra_disabled_collisions:
                at_check_links.append(ml)
        attach_check_pairs.append((at_check_links, attached.child))
    # * body pairs
    check_body_pairs = list(pp.interfaces.robots.product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    check_body_link_pairs = []
    for body1, body2 in check_body_pairs:
        body1, links1 = pp.interfaces.robots.expand_links(body1)
        body2, links2 = pp.interfaces.robots.expand_links(body2)
        if body1 == body2:
            continue
        bb_link_pairs = pp.interfaces.robots.product(links1, links2)
        for bb_links in bb_link_pairs:
            bbll_pair = ((body1, bb_links[0]), (body2, bb_links[1]))
            if bbll_pair not in extra_disabled_collisions and bbll_pair[::-1] not in extra_disabled_collisions:
                check_body_link_pairs.append(bbll_pair)
    # * joint limits
    lower_limits = env.joint_lower_limits
    upper_limits = env.joint_upper_limits

    # TODO: maybe prune the link adjacent to the robot
    def collision_fn(q, diagnosis=False):
        # * joint limit check
        if not all_between(lower_limits, q, upper_limits):
            return True
        # * set body & attachment positions
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()
        # * self-collision link check
        for link1, link2 in self_check_link_pairs:
            if pp.interfaces.robots.pairwise_link_collision(body, link1, body, link2):
                if diagnosis:
                    # warnings.warn('moving body link - moving body link collision!', UserWarning)
                    cr = pp.interfaces.robots.pairwise_link_collision_info(body, link1, body, link2)
                    draw_collision_diagnosis(cr, body_name_from_id=body_name_from_id, **kwargs)
                return True
        # * self link - attachment check
        for body_check_links, attached_body in attach_check_pairs:
            if pp.interfaces.robots.any_link_pair_collision(body, body_check_links, attached_body, **kwargs):
                if diagnosis:
                    # warnings.warn('moving body link - attachement collision!', UserWarning)
                    cr = pp.interfaces.robots.any_link_pair_collision_info(body, body_check_links, attached_body, **kwargs)
                    draw_collision_diagnosis(cr, body_name_from_id=body_name_from_id, **kwargs)
                return True
        # * body - body check
        for (body1, link1), (body2, link2) in check_body_link_pairs:
            if pp.interfaces.robots.pairwise_link_collision(body1, link1, body2, link2, **kwargs):
                if diagnosis:
                    # warnings.warn('moving body - body collision!', UserWarning)
                    cr = pp.interfaces.robots.pairwise_link_collision_info(body1, link1, body2, link2)
                    draw_collision_diagnosis(cr, body_name_from_id=body_name_from_id, **kwargs)
                return True
        return False
    return collision_fn


def plan_birrt(env,start,goal):
    # col_fn = pp.interfaces.planner_interface.joint_motion_planning.get_collision_fn(env.manipulator,env.joints)
    col_fn = get_collision_fn(env.manipulator,env.joints,env,obstacles=env.obs_ids)
    ext_fn = pp.interfaces.planner_interface.joint_motion_planning.get_extend_fn(env.manipulator,env.joints)
    d_fn = pp.interfaces.planner_interface.joint_motion_planning.get_distance_fn(env.manipulator,env.joints)
    sample_fn = get_sample_fn(env)


    plan = pp.motion_planners.birrt(start=start,goal=goal,collision_fn=col_fn,distance_fn=d_fn,sample_fn=sample_fn,extend_fn=ext_fn)

    return plan


